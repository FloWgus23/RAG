"""
hybrid_rag.py — Hybrid mode ของ Flowchat: ค้นทั้ง Vector และ Graph "พร้อมกัน" ในคำถามเดียว

ปัญหาที่แก้: ผู้ใช้ไม่รู้ว่าคำถามแบบไหนควรใช้โหมดไหน
    - "ฟังก์ชัน X ทำงานยังไง / เอกสารพูดถึงอะไร"   -> เหมาะกับ Vector (ค้นจากความหมายของเนื้อหา)
    - "ใครเรียก X / ถ้าแก้ X กระทบอะไร"             -> เหมาะกับ Graph (ค้นจากความสัมพันธ์ของโค้ด)
Hybrid ไม่ต้องเลือก: ดึงทั้งสองแหล่ง แล้วรวมเป็น context เดียวที่แยกส่วนชัดเจน ([GRAPH] / [VECTOR])
ให้ LLM เลือกใช้ส่วนที่ตรงกับคำถามเอง

หลักการ
- แต่ละแหล่งทำงานอิสระ: ถ้าแหล่งหนึ่งไม่พบ/ล้มเหลว อีกแหล่งยังทำงานต่อ (ไม่ล้มทั้งคำถาม)
- Graph จะมีผลลัพธ์ก็ต่อเมื่อคำถามมีชื่อฟังก์ชัน/คลาส/ไฟล์ที่อยู่ใน dataset — คำถามทั่วไปจึงได้แค่ Vector
- มีงบประมาณตัวอักษร (max_chars) รวมของ context: ภาษาไทยกิน token เยอะ ถ้า context ใหญ่เกิน num_ctx
  Ollama จะตัดส่วนต้นของ prompt ทิ้งเงียบๆ (ซึ่งเป็นที่อยู่ของ context) — Graph ได้ก่อนไม่เกิน 60% ที่เหลือให้ Vector
"""

from __future__ import annotations

from graph_rag import CodeGraph
from vector_rag import VectorStore

GRAPH_TAG = "🕸️"
VECTOR_TAG = "🔎"

GRAPH_HEADER = "=== [GRAPH] โครงสร้างโค้ด (ความสัมพันธ์ใครเรียกใคร / import) ==="
VECTOR_HEADER = "=== [VECTOR] เนื้อหาที่ใกล้ความหมายของคำถาม (ตัดจากเอกสาร/โค้ดต้นฉบับ) ==="

_GRAPH_SHARE = 0.6           # สัดส่วนงบสูงสุดของ Graph เมื่อมีผล Vector ด้วย
_MIN_PARTIAL = 400           # เหลืองบน้อยกว่านี้ไม่ยัดบางส่วน (ข้ามไปเลย ดีกว่าใส่ครึ่งๆ กลางๆ)
_TRUNC_NOTE = "\n…(ตัดเพื่อประหยัด context)"

HYBRID_SYSTEM_PROMPT = (
    "\n\nข้อมูลต่อไปนี้ดึงมาจากเอกสารและโค้ดที่ผู้ใช้อัปโหลดจริง ไม่ใช่จากความจำของคุณ "
    "มาจาก 2 แหล่งพร้อมกัน (อาจมีแหล่งเดียวก็ได้):\n"
    "- [GRAPH] = ความสัมพันธ์เชิงโครงสร้างของโค้ด (ใครเรียกใคร / import อะไร / ผลกระทบถ้าแก้) "
    "ใช้ยึดเป็นหลักเมื่อถามว่าใครเรียกใคร หรือถ้าแก้แล้วกระทบอะไร "
    "ใน [GRAPH] หัวข้อ (A) = สิ่งที่ node นั้นเรียกออกไปเอง, หัวข้อ (B) = สิ่งอื่นที่เรียกกลับมาหา node นั้น/ผลกระทบ "
    "สองหัวข้อนี้ทิศตรงข้ามกัน ห้ามสลับกัน\n"
    "- [VECTOR] = เนื้อหาต้นฉบับ (โค้ด/เอกสาร) ที่ใกล้ความหมายของคำถาม ใช้อธิบายว่าทำงานอย่างไร หรือเนื้อหาว่าอย่างไร\n"
    "ใช้เฉพาะส่วนที่เกี่ยวกับคำถาม ส่วนที่ไม่เกี่ยวให้ข้ามไปโดยไม่ต้องพูดถึง "
    "ถ้าสองส่วนขัดกันเรื่องความสัมพันธ์ ให้เชื่อ [GRAPH] "
    "ห้ามเสริมข้อมูลที่ไม่มีในเนื้อหานี้ รายการที่มี (?) คือยังไม่แน่ใจ ให้บอกผู้ใช้ว่าไม่แน่ใจ "
    "หากข้อมูลไม่พอต่อการตอบ ให้บอกตามตรงว่าไม่พบข้อมูลที่เกี่ยวข้อง "
    "ห้ามพูดถึงป้าย [GRAPH] หรือ [VECTOR] ในคำตอบ ให้อ้างเป็นชื่อไฟล์/ฟังก์ชันแทน "
    "และเมื่อตอบว่าใครเรียกใช้อะไร ให้แยก \"เรียกตรงๆ (ชั้น 1)\" ออกจาก \"ผลกระทบทางอ้อม (ชั้น 2 ขึ้นไป)\" ห้ามรวมเป็นชั้นเดียว:\n\n"
)


def hybrid_search(vector_store: VectorStore, code_graph: CodeGraph, query: str, *, top_k: int = 4, min_score: float = 0.3,
                  use_mmr: bool = True, mmr_lambda: float = 0.7, graph_depth: int = 3,
                  graph_max_nodes: int = 3) -> dict:
    """
    ค้นทั้งสองแหล่ง คืน {"graph": [...], "vector": [...], "errors": [...]}
    รูปแบบผลลัพธ์แต่ละรายการเหมือน VectorStore.search()/CodeGraph.search(): {"text", "source", ...}
    """
    out: dict = {"graph": [], "vector": [], "errors": []}
    if not code_graph.is_empty():
        try:
            out["graph"] = code_graph.search(query, top_k=graph_max_nodes, depth=graph_depth)
        except Exception as e:                       # แหล่งหนึ่งพังไม่ควรทำให้อีกแหล่งพังตาม
            out["errors"].append(f"Graph: {e}")
    if not vector_store.is_empty():
        try:
            out["vector"] = vector_store.search(
                query, top_k=top_k, min_score=min_score, use_mmr=use_mmr, mmr_lambda=mmr_lambda,
            )
        except Exception as e:
            out["errors"].append(f"Vector: {e}")
    return out


def build_context(res: dict, max_chars: int = 5000) -> dict:
    """
    รวมผลจากสองแหล่งเป็น context เดียวภายในงบ max_chars
    คืน {"context", "sources", "n_graph", "n_vector", "truncated"}
      - sources: ชื่อไฟล์ที่ติดป้ายแหล่ง เช่น "🕸️ app.py", "🔎 notes.md" (ไว้แสดง "อ้างอิงจาก")
      - n_graph / n_vector: จำนวนรายการที่ใส่เข้า context จริง (หลังตัดตามงบ)
      - truncated: True ถ้ามีรายการถูกตัด/ข้ามเพราะเกินงบ
    """
    parts: list[str] = []
    sources: list[str] = []
    counts = {"graph": 0, "vector": 0}
    state = {"used": 0, "truncated": False}

    def fill(kind: str, header: str, tag: str, budget: int):
        first = True
        for r in res.get(kind, []):
            block = f"[{r['source']}]\n{r['text']}"
            head_cost = len(header) + 2 if first else 0
            room = budget - state["used"] - head_cost - 2
            if len(block) > room:
                state["truncated"] = True
                if room < _MIN_PARTIAL:
                    continue
                block = block[: room - len(_TRUNC_NOTE)] + _TRUNC_NOTE
            if first:
                parts.append(header)
                state["used"] += head_cost
                first = False
            parts.append(block)
            state["used"] += len(block) + 2
            sources.append(f"{tag} {r['source']}")
            counts[kind] += 1

    graph_budget = int(max_chars * _GRAPH_SHARE) if res.get("vector") else max_chars
    fill("graph", GRAPH_HEADER, GRAPH_TAG, graph_budget)
    fill("vector", VECTOR_HEADER, VECTOR_TAG, max_chars)

    return {
        "context": "\n\n".join(parts),
        "sources": sources,
        "n_graph": counts["graph"],
        "n_vector": counts["vector"],
        "truncated": state["truncated"],
    }
