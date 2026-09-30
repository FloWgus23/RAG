# 💬 Flowchat (Hybrid RAG)

ระบบแชท AI ที่ทำงานบนเครื่อง 100% (Local) โดยใช้สถาปัตยกรรม **Hybrid RAG** ที่ผสานการทำงานระหว่าง **Vector Search** (ค้นหาความหมาย) และ **Graph Analysis** (สืบค้นโครงสร้างความสัมพันธ์ของโค้ด .py) เข้าด้วยกัน ผู้ใช้เลือกสลับโหมดได้เองผ่านหน้า UI

## 🛠 Tech Stack
- **Environment:** Python 3.10+ (pip + venv)
- **UI:** Streamlit
- **LLM & Embeddings:** Ollama (เช่น `qwen2.5-coder:7b` และ `nomic-embed-text`)
- **Vector Search:** NumPy (in-memory cosine similarity, ไม่ต้องพึ่ง FAISS)
- **Graph Analysis:** `ast` มาตรฐานของ Python (ไม่ต้องพึ่ง Kuzu/networkx)

## 📋 Prerequisites (สิ่งที่ต้องมีก่อนเริ่ม)
1. **[Python 3.10+](https://www.python.org/downloads/)**: สำหรับรันตัวแอป
2. **[Ollama](https://ollama.com/)**: สำหรับรัน AI Model ภายในเครื่อง

---

## 🚀 Setup & Installation (วิธีติดตั้ง)

**1. Clone the repository**
```bash
git clone https://github.com/FloWgus23/RAG.git
cd RAG
```

**2. ติดตั้ง Dependencies**
โปรเจกต์นี้ใช้ `pip` ในการจัดการแพ็กเกจ รันคำสั่งเดียวเพื่อติดตั้งไลบรารีทั้งหมดจาก `requirements.txt`:
```bash
pip install -r requirements.txt
```

**3. ดาวน์โหลด AI Models (Ollama)**
เปิด Terminal/PowerShell แล้วรันคำสั่งเพื่อโหลดโมเดลภาษาและโมเดลทำเวกเตอร์:
```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

> **หมายเหตุ:** ต้องรัน `ollama serve` ทิ้งไว้เบื้องหลังเสมอขณะใช้งานระบบ

---

## 💡 How to Use (วิธีใช้งาน)

**Step 1: เตรียมชุดข้อมูล (Dataset)**
โปรเจกต์นี้แถมโฟลเดอร์ `my_dataset/` มาให้แล้ว — เป็น **สำเนาโค้ดของ Flowchat เอง** (7 ไฟล์ `.py`) ใช้เป็นชุดข้อมูลตัวอย่างสำหรับทดสอบทันที หรือจะอัปโหลดไฟล์ `.py`/`.pdf`/`.docx`/`.txt`/`.md` ของคุณเองแทนก็ได้

**Step 2: Indexing อัตโนมัติ (ไม่ต้องรันสคริปต์แยก)**
ต่างจากระบบทั่วไปที่ต้องรันสคริปต์ index ต่างหากก่อนเปิดแอป — Flowchat จะสร้าง **Vector index** และ **Code graph** ให้อัตโนมัติทันทีที่อัปโหลดไฟล์ผ่านหน้า **Knowledge Base** หรือกล่องแนบไฟล์ในหน้าแชท ไม่มีขั้นตอนเตรียมฐานข้อมูลแยกต่างหาก

**Step 3: เปิดหน้าจอ UI**
```bash
streamlit run app.py
```
เมื่อเปิดขึ้นมาแล้ว อัปโหลดไฟล์ dataset → เลือกโหมด **🔎 Vector** หรือ **🕸️ Graph** ใน sidebar → พิมพ์คำถามได้เลย

---

## 📁 Project Structure

- **`app.py`** — หน้าจอติดต่อผู้ใช้ (Streamlit) ประกอบ UI และ flow การทำงานทั้งหมด
- **`config.py`** — ค่าคงที่และค่าเริ่มต้นทั้งหมดของระบบ
- **`ollama_client.py`** — จัดการ connection กับ Ollama (cache ไว้ใช้ซ้ำเพื่อประสิทธิภาพ)
- **`ui_style.py`** — CSS ธีมของแอป + ตัว render ข้อความแชท
- **`vector_rag.py`** — Vector mode: อ่านไฟล์ → ตัด chunk (แบบ ast-aware สำหรับ `.py`) → embedding → cosine similarity
- **`graph_rag.py`** — Graph mode: วิเคราะห์ไฟล์ `.py` ด้วย `ast` → สร้างกราฟ import/เรียกใช้ฟังก์ชัน
- **`file_utils.py`** — ฟังก์ชันช่วยจัดการไฟล์อัปโหลด ใช้ร่วมกันทั้งสองโหมด
- **`my_dataset/`** — โฟลเดอร์สำหรับวางไฟล์โค้ดต้นฉบับ (ค่าเริ่มต้นคือโค้ดของ Flowchat เอง)

---

## 🧠 Dataset ที่ดีช่วยให้ AI ตอบแม่นขึ้นได้อย่างไร

RAG ทำงานตามหลัก **"garbage in, garbage out"** — ตัว LLM ฉลาดแค่ไหนก็ตาม ถ้า context ที่ส่งเข้าไปไม่ดี คำตอบก็จะไม่ดีตาม

| ปัจจัยของ dataset | ผลกับคำตอบ |
|---|---|
| docstring ชัดเจนในทุกฟังก์ชัน | Vector mode ค้นด้วยความหมายของเนื้อหา ถ้าไม่มี docstring ต้องเดาจากโค้ดดิบ แม่นน้อยกว่า |
| ตั้งชื่อฟังก์ชัน/ไฟล์สื่อความหมาย | Graph mode จับคู่ node จากชื่อที่ปรากฏในคำถามตรงๆ ชื่อกำกวมจะหาไม่เจอ |
| ไฟล์ขนาดพอดี ไม่ยัดทุกอย่างไว้ไฟล์เดียว | ไฟล์ยาวเกินไปจะถูกตัด chunk เยอะ เสี่ยง context ปนกันข้ามเรื่อง |
| import กันอย่างมีเหตุผล | Graph mode อ่านความสัมพันธ์จาก import จริง ถ้าไม่ import กันเลยกราฟจะไม่มีอะไรให้ตอบ |

โฟลเดอร์ `my_dataset/` ที่แถมมาผ่านเกณฑ์ทั้ง 4 ข้อนี้อยู่แล้ว เพราะเป็นโค้ดของ Flowchat เองที่มี docstring ครบ ตั้งชื่อสื่อความหมาย และ import กันเป็นระบบ

## ⚙️ เทคนิคในโค้ดที่ช่วยให้ตอบดีขึ้น

- **Python-aware chunking** (`vector_rag.chunk_python_file`) — แบ่ง chunk ตามขอบเขตฟังก์ชัน/คลาสจริงด้วย `ast` แทนการตัดทุก 800 ตัวอักษร ทำให้แต่ละ chunk เป็นฟังก์ชันที่สมบูรณ์เสมอ
- **Embedding task prefix** (`vector_rag._query_prefix` / `_document_prefix`) — เติม `"search_query: "` / `"search_document: "` อัตโนมัติสำหรับโมเดลที่แนะนำแบบนี้ (เช่น `nomic-embed-text`) เพิ่มความแม่นยำการค้นหา
- **Batch graph finalize** — อัปโหลดหลายไฟล์พร้อมกันจะคำนวณความสัมพันธ์ข้ามไฟล์ครั้งเดียวตอนจบ แทนคำนวณซ้ำทุกไฟล์
- **Anti-hallucination system prompt** — กำกับ LLM ให้ตอบจากข้อมูลที่ให้มาเท่านั้น ห้ามเสริมเติมเอง และบอกตรงๆ เมื่อหาไม่เจอ

## 🎛️ ปรับ Settings ให้ตอบแม่นขึ้น

- **Temperature ต่ำ (0.1–0.3)** เมื่อใช้ RAG เพื่อให้โมเดลยึดตามข้อมูลที่ให้มากกว่าสร้างคำตอบใหม่
- **top-k พอดี** (ค่าเริ่มต้น 4 เหมาะกับ dataset ขนาดเล็ก-กลาง) มากไปจะมี context ปนไม่เกี่ยวข้อง น้อยไปอาจพลาดข้อมูลสำคัญ
- **เลือก embedding model ให้ตรงภาษา** — ถ้าเนื้อหา/คำถามเป็นไทยเป็นหลัก ลองเปลี่ยนเป็น `bge-m3` ในแท็บ Settings

---

## 📤 สำหรับส่งงาน (Google Form)
1. อัดคลิป YouTube (≤ 5 นาที): เปิด Streamlit → ถามคำถามทั้ง Vector และ Graph mode → โชว์ Mode/Sources → กด Clear Chat
2. zip โฟลเดอร์ `my_dataset/` → อัป Google Drive → ตั้งสิทธิ์ Anyone with the link → Viewer → ลิงก์ Dataset
3. zip ทั้งโปรเจกต์ (ไฟล์ `.py` ทั้งหมด + `requirements.txt` + `README.md`) → อัป Google Drive → ตั้งสิทธิ์เดียวกัน → ลิงก์ Source Code
4. นำ 3 ลิงก์ (YouTube, Dataset, Source Code) ไปกรอกใน Google Form
