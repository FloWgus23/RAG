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
    "mode": "vector",       # "vector" = ค้นจากความหมาย, "graph" = ค้นจากความสัมพันธ์ของโค้ด
    "top_k": 4,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.5,
    "num_ctx": 4096,
}

UPDATE_INTERVAL = 0.08  # วินาที — throttle การ re-render หน้าจอตอน stream คำตอบ

FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "py": "🐍"}
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "py"]

MODE_LABELS = {"vector": ("🔎", "Vector"), "graph": ("🕸️", "Graph")}
