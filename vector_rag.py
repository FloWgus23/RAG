"""
vector_rag.py — Vector mode ของ Flowchat (ค้นข้อมูลจาก "ความหมาย")
- อ่านไฟล์ (.pdf, .docx, .txt, .md, .py)
- ตัดข้อความเป็น chunk — ไฟล์ .py ใช้ ast แบ่งตามขอบเขตฟังก์ชัน/คลาส (ดู chunk_python_file)
  แทนการตัดทุก N ตัวอักษรแบบไม่สนใจโครงสร้าง กันไม่ให้ฟังก์ชันถูกตัดขาดกลางคัน ซึ่งทำให้
  embedding จับความหมายผิดและ LLM ได้ context ที่ไม่สมบูรณ์
- สร้าง embedding ผ่าน Ollama แบบ batch (เร็วกว่าทีละชิ้น) พร้อมเติม task prefix ตามที่
  โมเดล embedding แต่ละตัวแนะนำ (เช่น nomic-embed-text ต้องการ "search_query:"/"search_document:"
  นำหน้า) ซึ่งเพิ่มความแม่นยำของการค้นหาได้จริงตามเอกสารของแต่ละโมเดล
- เก็บ vector ที่ normalize ไว้ล่วงหน้า เพื่อให้ search เร็วขึ้น (ไม่ต้องคำนวณ norm ซ้ำทุกครั้ง)
- Score threshold: ตัด chunk ที่ cosine similarity ต่ำกว่าเกณฑ์ทิ้ง ถ้าไม่มีอะไรผ่านเกณฑ์จะคืน [] เพื่อให้
  app บอกผู้ใช้ตรงๆ ว่า "ไม่พบข้อมูลที่เกี่ยวข้อง" แทนการยัด context ที่ไม่เกี่ยวให้ LLM แล้วตอบมั่ว
- MMR (Maximal Marginal Relevance): เลือกผลลัพธ์ที่ทั้ง "ตรงคำถาม" และ "ไม่ซ้ำกันเอง" เพื่อให้ top-k
  ครอบคลุมหลายมุมมอง ไม่ใช่ 4 chunk ที่พูดเรื่องเดียวกันหมด

คู่กับ graph_rag.py ซึ่งเป็น Graph mode (ค้นข้อมูลจาก "ความสัมพันธ์" ของโค้ด .py)
"""

from __future__ import annotations

import ast
import io
import re
import numpy as np
import ollama


def _chunk_spans(text: str, chunk_size: int = 800, overlap: int = 150) -> list[tuple[int, str]]:
    """เหมือน chunk_text แต่คืน (ตำแหน่งเริ่มใน text ที่ strip แล้ว, ข้อความ) ไว้หาว่า chunk อยู่ใต้หัวข้อไหน"""
    text = text.strip()
    if not text:
        return []
    spans = []
    start, length = 0, len(text)
    step = max(chunk_size - overlap, 1)
    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end].strip()
        if piece:
            spans.append((start, piece))
        start += step
    return spans


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """ตัดข้อความยาวๆ ให้เป็นชิ้นเล็ก พร้อม overlap กันบริบทขาด (ใช้กับไฟล์ที่ไม่ใช่ .py หรือ .py ที่ parse ไม่ได้)"""
    return [p for _, p in _chunk_spans(text, chunk_size, overlap)]


_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)


def chunk_labels(filename: str, text: str, chunk_size: int = 800, overlap: int = 150) -> list[tuple[str, str]]:
    """
    ตัดไฟล์ที่ไม่ใช่ .py เป็น chunk แล้วคืน (label, piece) โดย label = "[ชื่อไฟล์ — หัวข้อ markdown ล่าสุด]"
    ใช้ label นำหน้า piece ตอน embed เพื่อให้ chunk ของ README บอกได้ว่าตัวเองมาจากไฟล์/หัวข้อไหน
    (เดิม chunk ของ .md ไม่มีข้อมูลนี้เลย ทำให้คำถาม "KMeans ..." ไปตกที่ README ทั่วไปแทน README ของ unsupervised)
    """
    stripped = text.strip()
    headings = [(m.start(), m.group(1)) for m in _HEADING_RE.finditer(stripped)]
    out = []
    for start, piece in _chunk_spans(text, chunk_size, overlap):
        title = ""
        for pos, h in headings:
            if pos <= start:
                title = h
            else:
                break
        out.append((f"[{filename} — {title}]" if title else f"[{filename}]", piece))
    return out


def chunk_python_file(filename: str, source: str, max_chunk: int = 1600) -> list[str]:
    """
    แบ่ง chunk ไฟล์ .py ตาม "ขอบเขตความหมาย" (import header, แต่ละฟังก์ชัน, แต่ละคลาส)
    แทนการตัดทุก 800 ตัวอักษรดื้อๆ เหตุผล:
    - แต่ละ chunk เป็นหน่วยที่สมบูรณ์ในตัวเอง (ทั้งฟังก์ชัน+docstring) ไม่ขาดตอนกลางฟังก์ชัน
    - embedding จับ "ความหมาย" ของฟังก์ชันนั้นได้ตรงกว่า ไม่ปนกับเนื้อหาฟังก์ชันข้างเคียง
    - ผลคือ LLM ได้ context ที่ครบและตรงประเด็นกว่า ตอบคำถามเกี่ยวกับโค้ดได้แม่นขึ้น
    ถ้าไฟล์ parse ไม่ผ่าน (syntax error) จะ fallback ไปใช้ chunk_text ตามปกติ
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text(source)

    chunks: list[str] = []

    # -- header: docstring ของโมดูล + import ทั้งหมด รวมเป็น chunk เดียว --
    header_parts = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        header_parts.append(f'"""{module_doc}"""')
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(source, node)
            if seg:
                header_parts.append(seg)
    if header_parts:
        chunks.append(f"[{filename} — module header]\n" + "\n".join(header_parts))

    # -- แต่ละฟังก์ชัน/คลาสระดับบนสุด เป็น chunk ของตัวเอง --
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(source, node)
            if not seg:
                continue
            label = f"[{filename} — {node.name}]"
            if len(seg) > max_chunk:
                # ฟังก์ชัน/คลาสยาวผิดปกติ ค่อย fallback ตัดย่อยด้วย chunk_text
                for piece in chunk_text(seg):
                    chunks.append(f"{label}\n{piece}")
            else:
                chunks.append(f"{label}\n{seg}")

    return chunks or chunk_text(source)


def read_file(uploaded_file) -> str:
    """อ่านเนื้อหาไฟล์ที่อัปโหลดผ่าน Streamlit (st.file_uploader) ออกมาเป็น string"""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    # .txt, .md, .py และไฟล์ข้อความอื่นๆ
    return data.decode("utf-8", errors="ignore")


# -- task prefix ตามคำแนะนำของแต่ละโมเดล embedding (เพิ่มความแม่นยำการค้นหาได้จริง) -----
def _query_prefix(embed_model: str) -> str:
    m = embed_model.lower()
    if "nomic-embed" in m:
        return "search_query: "
    if "mxbai-embed" in m:
        return "Represent this sentence for searching relevant passages: "
    return ""


def _document_prefix(embed_model: str) -> str:
    m = embed_model.lower()
    if "nomic-embed" in m:
        return "search_document: "
    return ""


class VectorStore:
    """
    Vector store อย่างง่ายเก็บใน memory
    - ใช้ ollama.Client ตัวเดียวที่ share connection (เร็วกว่าสร้างใหม่ทุกครั้ง)
    - embed แบบ batch ในคราวเดียวแทนการวน loop ทีละ chunk
    - เก็บ vector ที่ normalize ไว้แล้ว (self._normed) เพื่อไม่ต้องคำนวณ norm ซ้ำตอน search
    """

    def __init__(self, client: "ollama.Client", embed_model: str = "nomic-embed-text"):
        self.client = client
        self.embed_model = embed_model
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.vectors: np.ndarray | None = None
        self._normed: np.ndarray | None = None

    # -- embedding -----------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """พยายามยิง embedding ทีเดียวทั้งชุดก่อน (เร็วกว่า) ถ้า API/โมเดลไม่รองรับค่อย fallback ทีละอัน"""
        try:
            resp = self.client.embed(model=self.embed_model, input=texts)
            vecs = resp.get("embeddings")
            if vecs:
                return np.array(vecs, dtype=np.float32)
        except Exception:
            pass
        return np.vstack([self._embed_one(t) for t in texts])

    def _embed_one(self, text: str) -> np.ndarray:
        resp = self.client.embeddings(model=self.embed_model, prompt=text)
        return np.array(resp["embedding"], dtype=np.float32)

    def _rebuild_normed(self):
        if self.vectors is None or len(self.vectors) == 0:
            self._normed = None
            return
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8
        self._normed = self.vectors / norms

    # -- document management -------------------------------------------
    def add_document(self, filename: str, text: str) -> int:
        if filename.lower().endswith(".py"):
            pieces = chunk_python_file(filename, text)      # .py มี label [ไฟล์ — ชื่อ] ในตัว chunk อยู่แล้ว
            embed_texts = list(pieces)
        else:
            labeled = chunk_labels(filename, text)          # ไฟล์อื่น: ใส่ label ตอน embed เท่านั้น (ข้อความที่โชว์ยังเป็นต้นฉบับ)
            pieces = [p for _, p in labeled]
            embed_texts = [f"{lab}\n{p}" for lab, p in labeled]
        if not pieces:
            return 0

        doc_prefix = _document_prefix(self.embed_model)
        to_embed = [doc_prefix + p for p in embed_texts] if doc_prefix else embed_texts
        vecs = self._embed_batch(to_embed)

        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.chunks.extend(pieces)          # เก็บเนื้อหาต้นฉบับ (ไม่ใส่ prefix/label) ไว้โชว์ผู้ใช้
        self.sources.extend([filename] * len(pieces))
        self._rebuild_normed()
        return len(pieces)

    def remove_document(self, filename: str):
        keep = [i for i, s in enumerate(self.sources) if s != filename]
        if not keep:
            self.chunks, self.sources, self.vectors, self._normed = [], [], None, None
            return
        self.chunks = [self.chunks[i] for i in keep]
        self.sources = [self.sources[i] for i in keep]
        self.vectors = self.vectors[keep]
        self._rebuild_normed()

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.chunks) == 0

    def document_names(self) -> list[str]:
        seen = []
        for s in self.sources:
            if s not in seen:
                seen.append(s)
        return seen

    def chunk_counts(self) -> dict[str, int]:
        """นับจำนวน chunk ต่อไฟล์ในรอบเดียว (เร็วกว่าเรียก .count() วนต่อไฟล์ ซึ่งเป็น O(n) ต่อครั้ง)"""
        counts: dict[str, int] = {}
        for s in self.sources:
            counts[s] = counts.get(s, 0) + 1
        return counts

    # -- search ----------------------------------------------------------
    @staticmethod
    def _top_indices(sims: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
        """เลือก k ตัวที่ similarity สูงสุดจาก cand (index) แบบ O(n) แล้วเรียงเฉพาะ k ตัวนั้น"""
        if cand.size > k:
            part = np.argpartition(-sims[cand], k - 1)[:k]
            cand = cand[part]
        return cand[np.argsort(-sims[cand])]

    def _mmr(self, sims: np.ndarray, shortlist: np.ndarray, top_k: int, lam: float) -> np.ndarray:
        """
        Maximal Marginal Relevance: วนเลือกทีละชิ้นด้วยสูตร
            score = lam * sim(query, d) - (1 - lam) * max(sim(d, ชิ้นที่เลือกไปแล้ว))
        lam = 1.0 คือเน้นตรงคำถามล้วน (เท่ากับ top-k ธรรมดา), lam ต่ำลง = เน้นความหลากหลายมากขึ้น
        ใช้ vector ที่ normalize ไว้แล้ว จึงคำนวณ similarity ระหว่าง chunk ได้ด้วย dot product ล้วนๆ
        """
        rel = sims[shortlist]
        vecs = self._normed[shortlist]
        pair = vecs @ vecs.T                      # similarity ระหว่าง chunk ทุกคู่ใน shortlist
        selected = [int(np.argmax(rel))]          # ชิ้นแรก = ตรงคำถามที่สุดเสมอ
        max_to_selected = pair[selected[0]].copy()
        remaining = np.ones(len(shortlist), dtype=bool)
        remaining[selected[0]] = False
        while len(selected) < top_k and remaining.any():
            scores = lam * rel - (1.0 - lam) * max_to_selected
            scores[~remaining] = -np.inf
            best = int(np.argmax(scores))
            selected.append(best)
            remaining[best] = False
            max_to_selected = np.maximum(max_to_selected, pair[best])
        return shortlist[selected]

    # -- keyword boost ---------------------------------------------------
    _KW_STOP = frozenset({"the", "and", "for", "how", "what", "with", "this", "that", "from", "does", "use", "are", "who", "why"})

    @staticmethod
    def _norm_kw(t: str) -> str:
        return t.lower().replace("-", "").replace("_", "")

    def _keyword_boost(self, query: str, max_boost: float = 0.25, per_term: float = 0.12) -> np.ndarray:
        """
        embedding อย่างเดียวพลาดคำเฉพาะเวลาคำถามเป็นไทยแต่เอกสารเป็นอังกฤษ (เช่น "KMeans จัดกลุ่มยังไง")
        จึงดึงคำอังกฤษ/ชื่อเฉพาะในคำถามมาเทียบตรงๆ กับเนื้อหา chunk และชื่อไฟล์ แล้วบวกคะแนนให้ chunk ที่ตรง
        คำที่พบในหลาย chunk (เช่น "value") ได้น้ำหนักน้อยลง คำที่พบเฉพาะไม่กี่ chunk ได้มากกว่า
        """
        n = len(self.chunks)
        boost = np.zeros(n, dtype=np.float32)
        terms = {self._norm_kw(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", query)}
        terms = {t for t in terms if len(t) >= 3 and t not in self._KW_STOP}
        if not terms:
            return boost
        haystack = [self._norm_kw(c) + " " + self._norm_kw(s) for c, s in zip(self.chunks, self.sources)]
        for t in terms:
            hit = np.fromiter((t in h for h in haystack), dtype=bool, count=n)
            df = int(hit.sum())
            if df == 0:
                continue
            boost[hit] += per_term * (1.0 - df / n)
        return np.minimum(boost, max_boost)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.3,
               use_mmr: bool = True, mmr_lambda: float = 0.7, fetch_k: int | None = None):
        """
        ค้น chunk ที่ใกล้คำถามที่สุด
        - min_score: ตัดผลที่ cosine similarity ต่ำกว่าค่านี้ทิ้ง (ไม่มีอะไรผ่านเกณฑ์ -> คืน [])
          หมายเหตุ: สเกลคะแนนต่างกันตามโมเดล embedding ปรับได้ที่หน้า Settings
        - คะแนนรวม keyword boost แล้ว (ดู _keyword_boost) จึงอาจเกิน 1.0 เล็กน้อย
        - use_mmr / mmr_lambda: เลือกผลแบบหลากหลายด้วย MMR จาก shortlist (fetch_k ตัวที่ผ่านเกณฑ์)
        ผลลัพธ์แต่ละรายการมี "score" = similarity กับคำถามจริง (ไม่ใช่คะแนน MMR)
        """
        if self.is_empty():
            return []
        q_prefix = _query_prefix(self.embed_model)
        q = self._embed_one(q_prefix + query if q_prefix else query)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = self._normed @ qn  # ไม่ต้องคำนวณ norm ของ matrix ใหม่ทุกครั้ง
        sims = sims + self._keyword_boost(query)   # บวกคะแนนให้ chunk ที่มีคำอังกฤษ/ชื่อเฉพาะตรงกับคำถาม

        cand = np.flatnonzero(sims >= min_score)          # 1) ตัดทิ้งตาม score threshold
        if cand.size == 0:
            return []

        top_k = max(1, min(top_k, cand.size))
        if use_mmr and cand.size > top_k:
            fetch_k = max(fetch_k or top_k * 4, top_k)
            shortlist = self._top_indices(sims, cand, min(fetch_k, cand.size))   # 2) shortlist ตามความใกล้
            lam = min(max(mmr_lambda, 0.0), 1.0)
            idx = self._mmr(sims, shortlist, top_k, lam)                          # 3) เลือกให้หลากหลาย
        else:
            idx = self._top_indices(sims, cand, top_k)

        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(sims[i])}
            for i in idx
        ]
