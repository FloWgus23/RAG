# Flowchat — Local LLM Chat + Hybrid RAG (Vector + Graph)

แชทกับ LLM ที่รันบนเครื่องตัวเอง (Ollama) พร้อมระบบ **Hybrid RAG สองโหมด**:
- **🔎 Vector mode** — ค้นข้อมูลจาก "ความหมาย" (semantic search) เหมาะกับเอกสารทั่วไปและคำถามเชิงเนื้อหา
- **🕸️ Graph mode** — ค้นข้อมูลจาก "ความสัมพันธ์" ของโค้ด (import / เรียกใช้ฟังก์ชัน) เหมาะกับคำถามเชิงโครงสร้างโค้ด

งานวิชา Select Topic in Software

---

## 1. โครงสร้างโปรเจกต์

```
Flowchat/
├── app.py            — ประกอบ UI และ flow การทำงานทั้งหมด (entry point)
├── config.py         — ค่าคงที่และค่าเริ่มต้นทั้งหมด
├── ollama_client.py  — จัดการ connection กับ Ollama (cache ไว้ใช้ซ้ำ เพื่อประสิทธิภาพ)
├── ui_style.py        — CSS ธีมของแอป + ตัว render ข้อความแชท
├── vector_rag.py      — Vector mode: อ่านไฟล์ → ตัด chunk → embedding → cosine similarity
├── graph_rag.py       — Graph mode: วิเคราะห์ไฟล์ .py ด้วย ast → สร้างกราฟ import/เรียกใช้ฟังก์ชัน
├── file_utils.py      — ฟังก์ชันช่วยจัดการไฟล์อัปโหลด ใช้ร่วมกันทั้งสองโหมด
├── requirements.txt
├── README.md
└── my_dataset/         — ชุดข้อมูลตัวอย่างสำหรับสาธิตระบบ (ดูหัวข้อ 4)
```

แยกเป็นโมดูลเล็กๆ ที่มีหน้าที่ชัดเจน แทนการยัดทุกอย่างไว้ในไฟล์เดียว ทำให้:
- อ่าน/แก้ไขง่ายขึ้น แต่ละไฟล์รับผิดชอบเรื่องเดียว
- โมดูลต่างๆ import กันเป็นระบบ (`app.py` → `config`, `ollama_client`, `ui_style`, `vector_rag`, `graph_rag`, `file_utils`) ซึ่งเป็นตัวอย่างความสัมพันธ์ของโค้ดที่ Graph mode ใช้สาธิตได้ทันที

---

## 2. เตรียม Ollama

```bash
ollama serve
ollama pull <โมเดลแชทที่ต้องการ>     # เช่น qwen2.5-coder:7b หรือโมเดลอื่นที่มีในเครื่อง
ollama pull nomic-embed-text          # ใช้สำหรับ Vector mode เท่านั้น (Graph mode ไม่ต้องใช้)
```

## 3. ติดตั้งและรัน

```bash
pip install -r requirements.txt
streamlit run app.py
```

ไม่มี dependency เพิ่มสำหรับ Graph mode — ใช้ `ast` ซึ่งเป็นไลบรารีมาตรฐานของ Python เอง เบาและไม่ต้องพึ่งไลบรารีหนักอย่าง Kuzu/networkx

---

## 4. Dataset: `my_dataset/`

โฟลเดอร์นี้คือ **สำเนาโค้ดของ Flowchat เอง** (7 ไฟล์ `.py` ตามหัวข้อ 1) ใช้เป็นชุดข้อมูลตัวอย่างสำหรับสาธิตทั้งสองโหมด — พูดง่ายๆ คือ **Flowchat วิเคราะห์ตัวเอง**

เหตุผลที่เลือกใช้โค้ดตัวเอง:
- มีความสัมพันธ์ระหว่างไฟล์จริง (`app.py` import เกือบทุกโมดูล) ทำให้ Graph mode มีอะไรให้แสดงผล
- ไม่ต้องหา dataset จากภายนอก พิสูจน์ได้ทันทีว่าระบบทำงานถูกต้อง เพราะรู้คำตอบที่ถูกอยู่แล้ว

### วิธีใช้
1. ไปที่แท็บ **Knowledge Base** แล้วอัปโหลดไฟล์ทั้ง 7 ไฟล์จากโฟลเดอร์ `my_dataset/`
   (หรือกดเปิดกล่อง "📎 แนบไฟล์ความรู้เพิ่มเติม" ในหน้าแชทแล้วอัปโหลดได้เลยโดยไม่ต้องออกจากหน้าแชท)
2. ไฟล์ `.py` ทุกไฟล์จะถูก index เข้า **ทั้ง Vector store และ Code graph พร้อมกันอัตโนมัติ**
3. เปิดสวิตช์ "ใช้ RAG" ใน sidebar แล้วเลือกโหมด **Vector** หรือ **Graph** ตามที่ต้องการทดสอบ

### ตัวอย่างคำถามทดสอบ

**Vector mode** (ค้นจากความหมาย/เนื้อหา):
- "ฟังก์ชัน bubble_html ทำหน้าที่อะไร"
- "โปรเจกต์นี้ cache การเชื่อมต่อ Ollama ยังไง"
- "CSS ของแอปนี้ใช้สีหลักอะไร"

**Graph mode** (ค้นจากความสัมพันธ์/โครงสร้าง):
- "app.py import โมดูลอะไรบ้าง"
- "get_client ถูกเรียกใช้จากที่ไหนบ้าง"
- "ถ้าแก้ config.py จะกระทบไฟล์ไหนบ้าง"

คำตอบทั้งสองโหมดจะโชว์ **badge บอก Mode ที่ใช้** (🔎 Vector / 🕸️ Graph) และ **แหล่งอ้างอิง (Sources)** ต่อท้ายข้อความเสมอ

> ต้องการทดสอบกับ dataset อื่น (เช่นโค้ดโปรเจกต์ของคุณเอง) ก็อัปโหลดไฟล์ `.py` ชุดอื่นแทนได้เลย ระบบไม่ผูกกับไฟล์ชุดนี้ตายตัว

---

## 5. ฟีเจอร์อื่นๆ
- **Clear Chat** — ปุ่ม "🗑️ ล้างประวัติแชท" ใน sidebar ล้างบทสนทนาทั้งหมด (Vector store / Code graph ไม่ถูกล้างไปด้วย)
- **แสดงเวลาที่ใช้ตอบแบบ real-time** — ระหว่างตอบมี badge นับเวลาสด (⏳) พอตอบเสร็จเปลี่ยนเป็นเวลารวม (⚡)
- **ปรับพารามิเตอร์โมเดล** — Temperature และ num_ctx ปรับได้ในแท็บ Settings

## 6. ปรับแต่งเพิ่มเติม
- เปลี่ยนโมเดลแชท/embedding, system prompt, top-k ได้ในแท็บ **Settings**
- ปรับสีธีมได้ที่ CSS variables ด้านบนของ `ui_style.py` (ตัวแปร `--primary`, `--bg` ฯลฯ)
- Graph mode จับคู่ชื่อฟังก์ชันแบบ best-effort (ไม่ได้ resolve type เต็มรูปแบบ) เหมาะกับโค้ด dataset ขนาดเล็ก-กลาง

---

## 7. สำหรับส่งงาน (Google Form)
1. อัดคลิป YouTube (≤ 5 นาที) สาธิต: เปิด Streamlit → ถามคำถามทั้ง Vector และ Graph mode → โชว์ Mode/Sources → กด Clear Chat
2. zip โฟลเดอร์ `my_dataset/` → อัป Google Drive → ตั้งสิทธิ์ Anyone with the link → Viewer → นี่คือลิงก์ Dataset
3. zip ทั้งโปรเจกต์ (ไฟล์ `.py` ทั้งหมด + `requirements.txt` + `README.md`) → อัป Google Drive → ตั้งสิทธิ์เดียวกัน → นี่คือลิงก์ Source Code
4. นำ 3 ลิงก์ (YouTube, Dataset, Source Code) ไปกรอกใน Google Form
