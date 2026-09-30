# 💬 Flowchat (Hybrid RAG)

ระบบแชท AI ที่ทำงานบนเครื่อง 100% (Local) โดยใช้สถาปัตยกรรม **Hybrid RAG** ที่ผสานการทำงานระหว่าง **Vector Search** (ค้นหาความหมาย) และ **Graph Analysis** (สืบค้นโครงสร้างความสัมพันธ์ของโค้ด .py) เข้าด้วยกัน ผู้ใช้เลือกโหมดได้เองผ่านหน้า UI — หรือใช้ **🔀 Hybrid** (ค่าเริ่มต้น) ที่ค้นทั้งสองแบบพร้อมกันโดยไม่ต้องเดาว่าคำถามนี้ควรใช้โหมดไหน

---

## 🛠 Tech Stack
- **Environment:** Python 3.10+ (pip + venv)
- **UI:** Streamlit
- **LLM & Embeddings:** Ollama (เช่น `qwen2.5-coder:7b` และ `nomic-embed-text`)
- **Vector Search:** NumPy (in-memory cosine similarity, ไม่ต้องพึ่งไลบรารีภายนอกอย่าง FAISS)
- **Graph Analysis:** `ast` มาตรฐานของ Python (ไม่ต้องพึ่ง Kùzu หรือ NetworkX)

---

## 📋 Prerequisites (สิ่งที่ต้องมีก่อนเริ่ม)
1. **[Python 3.10+](https://www.python.org/downloads/)**: สำหรับรันแอปพลิเคชัน
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

> **หมายเหตุ:** ต้องเปิดโปรแกรม Ollama หรือรัน `ollama serve` ทิ้งไว้เบื้องหลังเสมอขณะใช้งานระบบ

---

## 💡 How to Use (วิธีใช้งาน)

**Step 1: เตรียมชุดข้อมูล (Dataset)**
Dataset ตัวอย่างของโปรเจกต์นี้คือโค้ดของ **[SmolML](https://github.com/rodmarkun/SmolML)** — ไลบรารี Machine Learning ที่เขียนด้วย Python ล้วนๆ (ใช้แค่ standard library) เพื่อการศึกษา ประกอบด้วย autograd (`Value`), อาร์เรย์หลายมิติ (`MLArray`), preprocessing, optimizers, loss functions, regression, neural network, tree models และ K-Means

รันสคริปต์เพื่อดึง dataset มาใส่โฟลเดอร์ `my_dataset/`:
```bash
python fetch_dataset.py                       # โหลด ZIP จาก GitHub อัตโนมัติ
python fetch_dataset.py --zip SmolML-main.zip # หรือระบุไฟล์ ZIP ที่ดาวน์โหลดไว้เอง
python fetch_dataset.py --with-tests          # (ไม่บังคับ) รวมโฟลเดอร์ tests/*.py
```
สคริปต์จะเก็บเฉพาะไฟล์ `.py` ใน `smolml/` และไฟล์ `.md` (README ของแต่ละส่วน) พร้อมตั้งชื่อไฟล์ตามลำดับแพ็กเกจ เช่น `smolml/core/ml_array.py` → `smolml.core.ml_array.py` เพื่อให้ Graph mode สามารถเชื่อมโยง import ข้ามไฟล์ได้อย่างแม่นยำ (สามารถนำไฟล์ `.py`, `.pdf`, `.docx`, `.txt`, `.md` อื่นๆ มาใช้งานแทนได้เช่นกัน)

**Step 2: Indexing อัตโนมัติ (ไม่ต้องรันสคริปต์แยก)**
ระบบจะสร้าง **Vector Index** และ **Code Graph** ให้อัตโนมัติทันทีที่อัปโหลดไฟล์ผ่านหน้า **Knowledge Base** หรือกล่องแนบไฟล์ในหน้าแชท โดยไม่ต้องรันสคริปต์เตรียมฐานข้อมูลแยกต่างหาก

**Step 3: เปิดหน้าจอ UI**
```bash
streamlit run app.py
```
เมื่อเปิดเบราว์เซอร์ขึ้นมา ให้อัปโหลดไฟล์ dataset → เลือกโหมดค้นหา **🔀 Hybrid** (แนะนำ) / **🔎 Vector** / **🕸️ Graph** ที่ sidebar → เริ่มพิมพ์คำถามได้ทันที

---

## 📁 Project Structure

- **`app.py`** — หน้าจอติดต่อผู้ใช้ (Streamlit) จัดการ UI และ flow การทำงานหลัก
- **`config.py`** — ค่าคงที่และการตั้งค่าเริ่มต้นทั้งหมดของระบบ
- **`ollama_client.py`** — จัดการ connection และ cache การเรียกใช้งาน Ollama
- **`ui_style.py`** — จัดการธีม CSS และเรนเดอร์กล่องข้อความแชท
- **`vector_rag.py`** — Vector mode: อ่านไฟล์ → ตัด chunk (ast-aware สำหรับ `.py`) → embedding → cosine similarity
- **`graph_rag.py`** — Graph mode: วิเคราะห์โครงสร้างโค้ดด้วย `ast` → สร้างกราฟ import/function call
- **`hybrid_rag.py`** — Hybrid mode: ผสานการทำงานของ Vector และ Graph พร้อมควบคุมสัดส่วน context
- **`file_utils.py`** — โมดูลช่วยจัดการการอ่านและประมวลผลไฟล์อัปโหลด
- **`fetch_dataset.py`** — สคริปต์ดาวน์โหลดและจัดรูปแบบชุดข้อมูล SmolML
- **`my_dataset/`** — โฟลเดอร์เก็บไฟล์ชุดข้อมูลที่พร้อมใช้งาน

---

## 🧠 Dataset ที่ดีช่วยให้ AI ตอบแม่นขึ้นได้อย่างไร

RAG ทำงานตามหลัก **"garbage in, garbage out"** — หากบริบท (context) ที่ส่งให้ LLM ขาดความชัดเจน คำตอบที่ได้ย่อมคลาดเคลื่อน

| ปัจจัยของ Dataset | ผลลัพธ์ต่อการทำงานของระบบ |
|---|---|
| **มี docstring ชัดเจนในทุกฟังก์ชัน** | Vector mode ค้นหาจากความหมายของข้อความ การมีคำอธิบายช่วยให้จับคู่ semantic ได้แม่นยำกว่าการอ่านโค้ดดิบ |
| **ตั้งชื่อตัวแปร/ฟังก์ชันสื่อความหมาย** | Graph mode จับคู่ node จากชื่อที่ปรากฏในคำถามโดยตรง หากตั้งชื่อกำกวมจะสืบค้นความสัมพันธ์ไม่พบ |
| **ขนาดไฟล์และโครงสร้างแบ่งเป็นสัดส่วน** | การแยกไฟล์ย่อยตามหน้าที่ช่วยลดการเกิด chunk ขนาดใหญ่เกินไป และป้องกันปัญหา context ปะปนกันข้ามโมดูล |
| **โครงสร้างการ import ชัดเจน** | Graph mode อ่านความสัมพันธ์จากการเรียกโมดูลจริง ทำให้แกะรอย dependency ข้ามระบบได้อย่างถูกต้อง |

---

## ⚙️ เทคนิคทางเทคนิคที่ช่วยเพิ่มประสิทธิภาพ

- **Python-aware chunking** (`vector_rag.chunk_python_file`) — ตัด chunk ตามขอบเขตฟังก์ชันและคลาสจริงด้วย `ast` แทนการตัดตามความยาวคงที่ ทำให้บริบทของแต่ละ chunk สมบูรณ์ในตัวเอง
- **Embedding task prefix** (`vector_rag._query_prefix` / `_document_prefix`) — ใส่ Prefix เช่น `"search_query: "` / `"search_document: "` ให้โดยอัตโนมัติเมื่อใช้โมเดลกลุ่ม Nomic เพื่อเพิ่มความแม่นยำในการค้นหา
- **Batch graph finalize** — ประมวลผลและเชื่อมโยงความสัมพันธ์ข้ามไฟล์รวดเดียวหลังจากอัปโหลดไฟล์ทั้งหมดเสร็จสิ้น แทนการประมวลผลซ้ำทีละไฟล์
- **Hybrid context allocation** (`hybrid_rag.py`) — รวมข้อความจากทั้ง Graph และ Vector ภายใต้งบตัวอักษรที่กำหนด (`config.HYBRID_CONTEXT_CHARS` โดย Graph ได้ไม่เกิน 60%) เพื่อไม่ให้เกินขีดจำกัด `num_ctx` ของโมเดล
- **Score threshold** (`vector_rag.VectorStore.search`) — ตัด chunk ที่มี cosine similarity ต่ำกว่าเกณฑ์ (ค่าเริ่มต้น 0.30) และแจ้งผู้ใช้ทันทีเมื่อไม่พบข้อมูล ช่วยลดการเกิด hallucination
- **MMR (Maximal Marginal Relevance)** (`vector_rag.VectorStore._mmr`) — คัดเลือก chunk ที่ตรงกับคำถามและมีความหลากหลาย ไม่ดึงเนื้อหาที่ซ้ำซ้อนกันมาแสดง
- **Name collision resolution** (`graph_rag.CodeGraph._resolve_call`) — แก้ไขปัญหาชื่อฟังก์ชันซ้ำกันโดยตรวจสอบจาก statement การ import, method ภายใต้คลาส (`self`) และ type hints
- **Multi-hop traversal** (`graph_rag.CodeGraph.traverse`) — ค้นหาผลกระทบของโค้ดแบบหลายระดับชั้น (BFS traversal) เพื่อวิเคราะห์ dependency chain เมื่อมีการแก้ไขฟังก์ชันหรือไฟล์

---

## 🎛️ ข้อแนะนำในการปรับแต่ง Settings

- **Temperature ต่ำ (0.1–0.3):** ช่วยให้คำตอบยึดตามเนื้อหาที่ค้นพบจริง ลดการแต่งข้อมูลเพิ่มเติม
- **Score threshold:** ปรับเพิ่ม (0.4–0.5) หากต้องการกรองข้อมูลให้ตรงประเด็นเข้มงวดยิ่งขึ้น หรือปรับลดลงหากระบบแจ้งว่าไม่พบข้อมูลบ่อยเกินไป
- **MMR λ:** ค่า 0.7 เหมาะสำหรับการเน้นความตรงประเด็นเป็นหลัก ปรับเป็น 0.5 เมื่อต้องการคำตอบที่ครอบคลุมเนื้อหาหลายแง่มุม
- **Graph depth:** กำหนดไว้ที่ 2–3 ชั้นเพื่อความรวดเร็วและความกระชับของ context
- **top-k:** ค่าเริ่มต้น 4 ชิ้น เหมาะสมกับ dataset ขนาดเล็กถึงปานกลาง

---

## 🎬 ตัวอย่างคำถามสำหรับสาธิต

### กรณีที่ 1: อัปโหลดชุดข้อมูล SmolML (`my_dataset/`)
| โหมด | ตัวอย่างคำถาม | ผลลัพธ์และสิ่งที่ระบบแสดงผล |
|---|---|---|
| 🔀 Hybrid | `Adam optimizer ทำงานยังไง และถ้าแก้จะกระทบอะไร` | แสดงทั้งคำอธิบายการทำงาน (Vector) และรายการฟังก์ชันที่ได้รับผลกระทบ (Graph) |
| 🔀 Hybrid | `SmolML คืออะไร` | ค้นหาและสรุปเนื้อหาจาก README ผ่าน Vector Search โดยอัตโนมัติ |
| 🕸️ Graph | `ถ้าแก้ Value จะกระทบอะไรบ้าง` | แสดงสายสัมพันธ์และการสืบทอด/เรียกใช้งานคลาส `Value` แบบ Multi-hop |
| 🕸️ Graph | `MLArray ถูกใครเรียกใช้` | แจกแจงโมดูลและคลาสอื่นๆ ภายในโปรเจกต์ที่นำ `MLArray` ไปใช้งาน |
| 🕸️ Graph | `ถ้าแก้ smolml.utils.losses จะกระทบไฟล์ไหนบ้าง` | แสดงรายชื่อไฟล์ที่ import โมดูล losses ย้อนกลับ |
| 🔎 Vector | `StandardScaler กับ MinMaxScaler ต่างกันยังไง` | ดึงเนื้อหาและ docstring มาเปรียบเทียบความแตกต่างอย่างชัดเจน |
| 🔎 Vector | `ราคาทองคำวันนี้เท่าไหร่` | ระบบแจ้งว่าไม่พบข้อมูลที่เกี่ยวข้อง (ไม่กุคำตอบเนื่องจากติด Score Threshold) |

### กรณีที่ 2: อัปโหลดซอร์สโค้ดของ Flowchat เอง
| โหมด | ตัวอย่างคำถาม | ผลลัพธ์และสิ่งที่ระบบแสดงผล |
|---|---|---|
| 🔀 Hybrid | `read_file อัปโหลดไฟล์ยังไง และถ้าแก้จะกระทบอะไร` | แสดงโค้ดการจัดการไฟล์พร้อมแผนผังฟังก์ชันที่เรียกใช้งาน |
| 🕸️ Graph | `ถ้าแก้ read_file จะกระทบอะไรบ้าง` | แสดงผลกระทบแบบ 2 ชั้น: `process_uploaded_files` → `app` |
| 🕸️ Graph | `VectorStore.search ถูกใครเรียก` | ระบุคลาสและตำแหน่งที่เรียกใช้งานอย่างเจาะจง (ป้องกัน Name Collision) |
| 🕸️️ Graph | `ถ้าแก้ config.py จะกระทบไฟล์ไหนบ้าง` | แสดงรายชื่อไฟล์ทั้งหมดที่ import config ไปใช้งาน (`app`, `file_utils`, `ollama_client`) |