"""
config.py — ค่าคงที่และค่าเริ่มต้นทั้งหมดของ Flowchat
แยกออกมาเป็นไฟล์เดียว เพื่อให้โมดูลอื่น (app.py, ollama_client.py, ui_style.py, file_utils.py)
import มาใช้ร่วมกันได้ ไม่ต้อง hardcode ค่าซ้ำหลายที่
"""

APP_NAME = "Flowchat"

# ใช้กรองชื่อโมเดลที่เป็น embedding ออกจาก dropdown "โมเดลแชท" กันเลือกผิดแล้วเรียก chat ไม่ได้
EMBED_HINTS = ("embed", "minilm", "bge", "e5-")

DEFAULT_SYSTEM_PROMPT = "คุณเป็นผู้ช่วย AI ที่ตอบคำถามอย่างกระชับ ชัดเจน และเป็นมิตร"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# ค่าเริ่มต้นของ st.session_state ทั้งหมด รวมไว้ที่เดียวกันเพื่อไม่ให้ตกหล่น
SESSION_DEFAULTS = {
    "page": "chat",
    "messages": [],
    "chat_model": None,
    "embed_model": DEFAULT_EMBED_MODEL,
    "use_rag": True,
    "mode": "hybrid",       # "hybrid" = ทั้งสองพร้อมกัน (ค่าเริ่มต้น), "vector" = ค้นจากความหมาย, "graph" = ความสัมพันธ์ของโค้ด
    "top_k": 4,
    "min_score": 0.3,       # Vector: ตัด chunk ที่ cosine similarity ต่ำกว่านี้ทิ้ง (ไม่ผ่าน = "ไม่พบข้อมูลที่เกี่ยวข้อง")
    "use_mmr": True,        # Vector: เลือกผลแบบหลากหลาย (MMR) แทน top-k ที่ใกล้ที่สุดอย่างเดียว
    "mmr_lambda": 0.7,      # 1.0 = เน้นตรงคำถามล้วน, ต่ำลง = เน้นหลากหลายขึ้น
    "graph_depth": 3,       # Graph: ไล่ความสัมพันธ์ได้กี่ชั้น (multi-hop) เช่น ถ้าแก้ A กระทบใครต่อกี่ทอด
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.5,
    "num_ctx": 8192,
}

# Hybrid mode: งบตัวอักษรรวมของ context (Vector + Graph) และจำนวน code node สูงสุดจาก Graph
# ภาษาไทยกิน token เยอะ ถ้า context ใหญ่เกิน num_ctx Ollama จะตัดส่วนต้น prompt ทิ้งเงียบๆ — ถ้าเพิ่มค่านี้ ให้เพิ่ม num_ctx ด้วย
HYBRID_CONTEXT_CHARS = 5000
HYBRID_GRAPH_NODES = 3

# จำนวนข้อความล่าสุดที่ส่งให้โมเดลต่อครั้ง (1 คู่ถาม-ตอบ = 2 ข้อความ) กันประวัติแชทยาวไปเบียดที่ของ context
MAX_HISTORY_MESSAGES = 6
# งบตัวอักษรของประวัติแชทที่ส่งให้โมเดล (ไม่รวมข้อความล่าสุดของผู้ใช้ ซึ่งส่งครบเสมอ) และความยาวสูงสุดของคำตอบเก่าแต่ละข้อความ
# ภาษาไทยกิน token เยอะ คำตอบเก่ายาวๆ จะเบียดที่ของข้อมูลอ้างอิง (RAG) จน Ollama ตัดส่วนต้น prompt ทิ้ง
HISTORY_MAX_CHARS = 1500
HISTORY_OLD_MSG_CHARS = 400

UPDATE_INTERVAL = 0.08  # วินาที — throttle การ re-render หน้าจอตอน stream คำตอบ

FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "py": "🐍"}
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "py"]

MODE_LABELS = {"hybrid": ("🔀", "Hybrid"), "vector": ("🔎", "Vector"), "graph": ("🕸️", "Graph")}
