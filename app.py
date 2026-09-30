"""
app.py — Flowchat: LLM Chat App (Ollama + Hybrid RAG) บน Streamlit
งานวิชา Select Topic in Software

รัน: streamlit run app.py
ต้องมี Ollama รันอยู่ที่เครื่อง (ollama serve) และดึงโมเดลไว้แล้ว เช่น:
    ollama pull <โมเดลแชทที่ต้องการ>
    ollama pull nomic-embed-text   (สำหรับ Vector mode)

โครงสร้างโปรเจกต์ (แยกโมดูลเพื่อความเป็นระเบียบและอ่านง่าย):
    app.py            — ไฟล์นี้: ประกอบ UI และ flow การทำงานทั้งหมด
    config.py         — ค่าคงที่และค่าเริ่มต้นทั้งหมด
    ollama_client.py  — จัดการ connection กับ Ollama (cache ไว้ใช้ซ้ำ)
    ui_style.py       — CSS ธีมของแอป + ตัว render ข้อความแชท
    vector_rag.py     — Vector mode: ค้นข้อมูลจากความหมาย (embedding + cosine similarity)
    graph_rag.py      — Graph mode: ค้นข้อมูลจากความสัมพันธ์ของโค้ด (ast + import/call graph)
    file_utils.py     — ฟังก์ชันช่วยจัดการไฟล์อัปโหลด ใช้ร่วมกันทั้งสองโหมด

my_dataset/ ในโปรเจกต์นี้คือ "สำเนาโค้ดของ Flowchat เอง" ใช้เป็นชุดข้อมูลตัวอย่างสำหรับสาธิต
ทั้ง Vector mode (ถามความหมาย) และ Graph mode (ถามความสัมพันธ์ระหว่างไฟล์/ฟังก์ชัน)
"""

import time
from datetime import datetime

import streamlit as st

from config import APP_NAME, SESSION_DEFAULTS, SUPPORTED_TYPES
from ollama_client import get_client, get_ollama_models, chat_models_only
from ui_style import inject_css, bubble_html
from vector_rag import VectorStore
from graph_rag import CodeGraph
from file_utils import file_icon, process_uploaded_files, remove_document_everywhere

# ---------------------------------------------------------------------------
# Page config + ธีม
# ---------------------------------------------------------------------------
st.set_page_config(page_title=f"{APP_NAME} — AI Chat", page_icon="💬", layout="wide",
                    initial_sidebar_state="expanded")
inject_css()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
client = get_client()

for k, v in SESSION_DEFAULTS.items():
    st.session_state.setdefault(k, v)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(client, embed_model=st.session_state.embed_model)
else:
    st.session_state.vector_store.embed_model = st.session_state.embed_model

if "code_graph" not in st.session_state:
    st.session_state.code_graph = CodeGraph()

vector_store: VectorStore = st.session_state.vector_store
code_graph: CodeGraph = st.session_state.code_graph

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="brand"><div class="brand-badge">💬</div> {APP_NAME}</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-label">เมนู</div>', unsafe_allow_html=True)
    for key, label in [("chat", "💬  AI Chat"), ("docs", "📚  Knowledge Base"), ("settings", "⚙️  Settings")]:
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">โมเดลแชท</div>', unsafe_allow_html=True)

    all_models = get_ollama_models(client)
    chat_models = chat_models_only(all_models)

    if not all_models:
        st.warning("ต่อ Ollama ไม่ได้\nกรุณาเช็คว่ารัน `ollama serve` แล้ว", icon="⚠️")
    else:
        if st.session_state.chat_model not in chat_models:
            st.session_state.chat_model = chat_models[0]
        st.session_state.chat_model = st.selectbox(
            "โมเดลแชท", chat_models, index=chat_models.index(st.session_state.chat_model),
            label_visibility="collapsed",
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.session_state.use_rag = st.toggle("ใช้ RAG", value=st.session_state.use_rag,
                                          help="ดึงข้อมูลจากไฟล์ใน Knowledge Base มาช่วยตอบ")

    if st.session_state.use_rag:
        mode_options = {"vector": "🔎 Vector (ความหมาย)", "graph": "🕸️ Graph (ความสัมพันธ์โค้ด)"}
        current = st.session_state.mode
        chosen_label = st.radio(
            "โหมดค้นหา", list(mode_options.values()),
            index=list(mode_options.keys()).index(current),
            label_visibility="collapsed",
        )
        st.session_state.mode = next(k for k, v in mode_options.items() if v == chosen_label)

    n_docs = len(vector_store.document_names())
    n_nodes = code_graph.node_count()
    ok = bool(all_models)
    st.markdown(
        f'<div class="sidebar-stats">'
        f'📚 เอกสารในระบบ: <b>{n_docs}</b> ไฟล์ · 🕸️ code nodes: <b>{n_nodes}</b><br>'
        f'<span class="status-dot {"ok" if ok else "off"}"></span>Ollama '
        f'{"เชื่อมต่อแล้ว" if ok else "ยังไม่เชื่อมต่อ"}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if st.button("🗑️  ล้างประวัติแชท", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Knowledge Base
# ---------------------------------------------------------------------------
if st.session_state.page == "docs":
    st.markdown(
        '<div class="hero"><div class="hero-badge">📚</div><h1>Knowledge Base</h1>'
        '<p>อัปโหลดเอกสารหรือไฟล์โค้ด .py — ไฟล์ .py จะถูกวิเคราะห์เข้าทั้ง Vector และ Graph mode</p></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "อัปโหลดไฟล์ (.pdf, .docx, .txt, .md, .py) — เลือกได้หลายไฟล์พร้อมกัน",
        type=SUPPORTED_TYPES, accept_multiple_files=True, key="docs_page_uploader",
    )
    process_uploaded_files(uploaded, vector_store, code_graph)

    st.markdown('<div class="section-title">เอกสารในระบบ</div>', unsafe_allow_html=True)
    docs = vector_store.document_names()
    if not docs:
        st.caption("ยังไม่มีเอกสาร — อัปโหลดไฟล์ด้านบนเพื่อเริ่มต้น (แนะนำลองอัปโฟลเดอร์ my_dataset/ ที่แถมมากับโปรเจกต์)")
    else:
        chunk_counts = vector_store.chunk_counts()  # นับครั้งเดียว O(n) แทนการ .count() วนต่อไฟล์ O(n²)
        for d in docs:
            n_chunks = chunk_counts.get(d, 0)
            extra = f" · {code_graph.node_count(d)} code nodes" if d.endswith(".py") else ""
            c1, c2 = st.columns([8, 1])
            with c1:
                st.markdown(
                    f'<div class="doc-card"><div class="doc-icon">{file_icon(d)}</div>'
                    f'<div><div class="doc-name">{d}</div><div class="doc-meta">{n_chunks} chunks{extra}</div></div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("ลบ", key=f"del_{d}"):
                    remove_document_everywhere(d, vector_store, code_graph)
                    st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Settings
# ---------------------------------------------------------------------------
elif st.session_state.page == "settings":
    st.markdown(
        '<div class="hero"><div class="hero-badge">⚙️</div><h1>Settings</h1>'
        '<p>ปรับแต่งพฤติกรรมของ AI และการค้นหาข้อมูล</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">บุคลิกของ AI</div>', unsafe_allow_html=True)
    st.session_state.system_prompt = st.text_area("System prompt", st.session_state.system_prompt, height=100,
                                                   label_visibility="collapsed")

    st.markdown('<div class="section-title">การค้นหาข้อมูล (Vector mode)</div>', unsafe_allow_html=True)
    st.session_state.embed_model = st.text_input("Embedding model (Ollama)", st.session_state.embed_model)
    st.session_state.top_k = st.slider("จำนวน chunk / code node ที่ดึงมาอ้างอิง (top-k)", 1, 10, st.session_state.top_k)
    st.caption("ต้อง `ollama pull` โมเดลที่ระบุไว้แล้วในเครื่องก่อนใช้งาน · Graph mode ไม่ต้องใช้ embedding model")

    st.markdown('<div class="section-title">พารามิเตอร์การตอบของโมเดล</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 1.5, st.session_state.temperature, 0.05,
            help="ต่ำ = ตอบนิ่ง แม่นยำ ทำตามข้อมูล/คำสั่งเป๊ะ · สูง = ตอบสร้างสรรค์ หลากหลาย แต่เสี่ยงหลุดประเด็นมากขึ้น",
        )
    with c2:
        st.session_state.num_ctx = st.select_slider(
            "Context window (num_ctx)", options=[2048, 4096, 8192, 16384, 32768],
            value=st.session_state.num_ctx,
            help="ยิ่งมาก ยิ่งจำบทสนทนา/เอกสารอ้างอิงได้ยาวขึ้น แต่กินแรมและช้าลง",
        )
    st.caption(f"ค่าปัจจุบัน: temperature = {st.session_state.temperature:.2f}, "
               f"num_ctx = {st.session_state.num_ctx:,} tokens")

# ---------------------------------------------------------------------------
# PAGE: Chat
# ---------------------------------------------------------------------------
else:
    st.markdown(
        f'<div class="hero"><div class="hero-badge">💬</div><h1>Welcome to {APP_NAME}</h1>'
        '<p>แชทกับ LLM ที่รันบนเครื่องคุณเอง ผ่าน Ollama — Hybrid RAG: Vector + Graph mode</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("📎 แนบไฟล์ความรู้เพิ่มเติมสำหรับแชทนี้ (รองรับ .py สำหรับ Graph mode)"):
        chat_uploaded = st.file_uploader(
            "ลากไฟล์มาวาง หรือเลือกไฟล์ (.pdf, .docx, .txt, .md, .py)",
            type=SUPPORTED_TYPES, accept_multiple_files=True,
            key="chat_inline_uploader", label_visibility="collapsed",
        )
        process_uploaded_files(chat_uploaded, vector_store, code_graph)
        docs = vector_store.document_names()
        if docs:
            st.markdown(
                '<div class="attach-hint">📎 แนบอยู่: ' + ", ".join(f"{file_icon(d)} {d}" for d in docs) + '</div>',
                unsafe_allow_html=True,
            )

    history_ph = st.container()

    def render_history():
        with history_ph:
            for msg in st.session_state.messages:
                st.markdown(
                    bubble_html(msg["role"], msg["content"], msg.get("ts", ""),
                                msg.get("elapsed"), mode=msg.get("mode")),
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    src = ", ".join(sorted(set(msg["sources"])))
                    st.markdown(f'<div class="sources">📎 อ้างอิงจาก: {src}</div>', unsafe_allow_html=True)

    render_history()

    prompt = st.chat_input("พิมพ์ข้อความของคุณ...")

    if prompt:
        now = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "user", "content": prompt, "ts": now})
        with history_ph:
            st.markdown(bubble_html("user", prompt, now), unsafe_allow_html=True)

        # -- RAG retrieval: เลือกใช้ Vector หรือ Graph ตามโหมดที่ตั้งไว้ -----------
        context_block, used_sources = "", []
        active_mode = st.session_state.mode if st.session_state.use_rag else None

        if st.session_state.use_rag:
            try:
                if st.session_state.mode == "graph":
                    results = code_graph.search(prompt, top_k=st.session_state.top_k)
                    if not results:
                        st.info("🕸️ Graph mode: ไม่พบฟังก์ชัน/คลาส/โมดูลที่ตรงกับคำถามในกราฟโค้ด "
                                "— ลองระบุชื่อฟังก์ชัน/ไฟล์ให้ตรงกับ dataset หรือสลับไปโหมด Vector")
                else:
                    if vector_store.is_empty():
                        st.info("🔎 Vector mode: ยังไม่มีเอกสารใน Knowledge Base "
                                "— เปิดกล่อง 📎 ด้านบนหรือไปที่แท็บ Knowledge Base เพื่ออัปโหลดไฟล์ก่อน")
                    results = vector_store.search(prompt, top_k=st.session_state.top_k)

                if results:
                    context_block = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
                    used_sources = [r["source"] for r in results]
            except Exception as e:
                st.warning(f"ค้นหาข้อมูลไม่สำเร็จ: {e}")

        system_content = st.session_state.system_prompt
        if context_block:
            if st.session_state.mode == "graph":
                system_content += (
                    "\n\nข้อมูลต่อไปนี้คือความสัมพันธ์เชิงโครงสร้างของโค้ด (import / เรียกใช้ฟังก์ชัน) "
                    "ที่ดึงมาจากกราฟโค้ดจริง ไม่ใช่จากความจำของคุณ ใช้ข้อมูลนี้ตอบคำถามให้ตรงประเด็น "
                    "อ้างชื่อไฟล์/ฟังก์ชันให้ตรงกับที่ปรากฏ ห้ามเดาความสัมพันธ์ที่ไม่มีในข้อมูลนี้ "
                    "หากข้อมูลไม่พอต่อการตอบให้บอกตามตรงว่าไม่พบในกราฟโค้ดนี้:\n\n" + context_block
                )
            else:
                system_content += (
                    "\n\nข้อมูลต่อไปนี้ดึงมาจากเอกสารที่ผู้ใช้อัปโหลดจริง ให้ใช้เป็นหลักในการตอบหากเกี่ยวข้องกับคำถาม "
                    "ตอบให้ตรงกับเนื้อหาที่ให้มา ห้ามเสริมข้อมูลที่ไม่มีในเนื้อหานี้ขึ้นมาเอง "
                    "หากเนื้อหาที่ให้มาไม่เพียงพอต่อการตอบ ให้บอกตามตรงว่าไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร "
                    "แทนที่จะตอบจากความรู้ทั่วไป:\n\n" + context_block
                )

        ollama_messages = [{"role": "system", "content": system_content}]
        ollama_messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # -- Streaming response พร้อมนับเวลาสด -----------------------------
        answer = ""
        start = time.perf_counter()
        placeholder = st.empty()
        placeholder.markdown(bubble_html("assistant", "", elapsed=0.0, thinking=True, mode=active_mode),
                              unsafe_allow_html=True)

        if not st.session_state.chat_model:
            answer = "⚠️ ไม่พบโมเดลใน Ollama กรุณา `ollama pull <model>` ก่อนใช้งาน"
        else:
            try:
                last_update = 0.0
                stream = client.chat(
                    model=st.session_state.chat_model, messages=ollama_messages, stream=True,
                    options={"temperature": st.session_state.temperature, "num_ctx": st.session_state.num_ctx},
                )
                for chunk in stream:
                    answer += chunk["message"]["content"]
                    now_t = time.perf_counter()
                    if now_t - last_update >= 0.08:
                        placeholder.markdown(
                            bubble_html("assistant", answer, elapsed=now_t - start, thinking=True, mode=active_mode),
                            unsafe_allow_html=True,
                        )
                        last_update = now_t
            except Exception as e:
                answer = f"⚠️ เชื่อมต่อ Ollama ไม่สำเร็จ: {e}"

        total_elapsed = time.perf_counter() - start
        now = datetime.now().strftime("%H:%M")
        placeholder.markdown(bubble_html("assistant", answer, now, elapsed=total_elapsed, mode=active_mode),
                              unsafe_allow_html=True)
        if used_sources:
            src = ", ".join(sorted(set(used_sources)))
            st.markdown(f'<div class="sources">📎 อ้างอิงจาก: {src}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "ts": now, "elapsed": total_elapsed,
            "sources": used_sources, "mode": active_mode,
        })
