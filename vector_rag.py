"""
vector_rag.py — Vector mode ของ Flowchat (ค้นข้อมูลจาก "ความหมาย")
- อ่านไฟล์ (.pdf, .docx, .txt, .md, .py)
- ตัดข้อความเป็น chunk
- สร้าง embedding ผ่าน Ollama แบบ batch (เร็วกว่าทีละชิ้น)
- เก็บ vector ที่ normalize ไว้ล่วงหน้า เพื่อให้ search เร็วขึ้น (ไม่ต้องคำนวณ norm ซ้ำทุกครั้ง)

คู่กับ graph_rag.py ซึ่งเป็น Graph mode (ค้นข้อมูลจาก "ความสัมพันธ์" ของโค้ด .py)
"""

from __future__ import annotations

import io
import numpy as np
import ollama


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """ตัดข้อความยาวๆ ให้เป็นชิ้นเล็ก พร้อม overlap กันบริบทขาด"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start, length = 0, len(text)
    step = max(chunk_size - overlap, 1)
    while start < length:
        end = min(start + chunk_size, length)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


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
        pieces = chunk_text(text)
        if not pieces:
            return 0
        vecs = self._embed_batch(pieces)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.chunks.extend(pieces)
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

    # -- search ----------------------------------------------------------
    def search(self, query: str, top_k: int = 4):
        if self.is_empty():
            return []
        q = self._embed_one(query)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = self._normed @ qn  # ไม่ต้องคำนวณ norm ของ matrix ใหม่ทุกครั้ง
        top_k = min(top_k, len(self.chunks))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]     # หา top-k แบบ O(n) ก่อน
        idx = idx[np.argsort(-sims[idx])]                    # ค่อยเรียงแค่ top-k
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(sims[i])}
            for i in idx
        ]
