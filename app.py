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
    vector_rag.py     — Vector mode: ค้นข้อมูลจากความหมาย (embedding + cosine similarity + score threshold + MMR)
    graph_rag.py      — Graph mode: ค้นข้อมูลจากความสัมพันธ์ของโค้ด (ast + import/call graph, multi-hop, resolve ชื่อผ่าน import)
    file_utils.py     — ฟังก์ชันช่วยจัดการไฟล์อัปโหลด ใช้ร่วมกันทั้งสองโหมด

my_dataset/ ในโปรเจกต์นี้คือ "สำเนาโค้ดของ Flowchat เอง" ใช้เป็นชุดข้อมูลตัวอย่างสำหรับสาธิต
ทั้ง Vector mode (ถามความหมาย) และ Graph mode (ถามความสัมพันธ์ระหว่างไฟล์/ฟังก์ชัน)
"""

import queue
import re
import threading
import time
from datetime import datetime

import streamlit as st

from config import (APP_NAME, SESSION_DEFAULTS, SUPPORTED_TYPES, HYBRID_CONTEXT_CHARS, HYBRID_GRAPH_NODES,
                    MAX_HISTORY_MESSAGES, UPDATE_INTERVAL, HISTORY_MAX_CHARS, HISTORY_OLD_MSG_CHARS)
from hybrid_rag import hybrid_search, build_context, HYBRID_SYSTEM_PROMPT
from ollama_client import get_client, get_ollama_models, chat_models_only
from ui_style import inject_css, bubble_html, status_html
from vector_rag import VectorStore
from graph_rag import CodeGraph
from file_utils import file_icon, process_uploaded_files, remove_document_everywhere


def build_history(messages, max_messages=MAX_HISTORY_MESSAGES, max_chars=HISTORY_MAX_CHARS,
                  old_msg_chars=HISTORY_OLD_MSG_CHARS):
    """
    เลือกประวัติแชทที่จะส่งให้โมเดล โดยกันไม่ให้ประวัติเบียดข้อมูลอ้างอิง (RAG) จน context เกิน num_ctx
      - ข้อความล่าสุด (คำถามปัจจุบัน) ส่งครบเสมอ
      - ข้อความเก่าถูกตัดให้สั้นลง และหยุดเมื่อรวมเกินงบ max_chars
      - ลบหัวข้อ **[GRAPH]** / **[VECTOR]** ออกจากคำตอบเก่า กันโมเดลลอกรูปแบบมาใช้ในโหมดอื่น
    """
    tail = messages[-max_messages:]
    if not tail:
        return []
    out = [{"role": tail[-1]["role"], "content": tail[-1]["content"]}]
    used = 0
    for m in reversed(tail[:-1]):
        text = re.sub(r"\*\*\[(?:GRAPH|VECTOR)\]\*\*:?\s*", "", m["content"]).strip()
        if len(text) > old_msg_chars:
            text = text[:old_msg_chars] + "…"
        if used + len(text) > max_chars:
            break
        used += len(text)
        out.insert(0, {"role": m["role"], "content": text})
    return out

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
        mode_options = {"hybrid": "🔀 Hybrid (Vector + Graph)", "vector": "🔎 Vector (ความหมาย)",
                        "graph": "🕸️ Graph (ความสัมพันธ์โค้ด)"}
        current = st.session_state.mode
        chosen_label = st.radio(
            "โหมดค้นหา", list(mode_options.values()),
            index=list(mode_options.keys()).index(current),
            label_visibility="collapsed",
        )
        st.session_state.mode = next(k for k, v in mode_options.items() if v == chosen_label)
        if st.session_state.mode == "hybrid":
            st.caption("ค้นทั้งสองแบบพร้อมกัน ไม่ต้องเลือกเอง")

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
    st.session_state.min_score = st.slider(
        "เกณฑ์ความเกี่ยวข้องขั้นต่ำ (score threshold)", 0.0, 0.9, st.session_state.min_score, 0.05,
        help="chunk ที่ cosine similarity ต่ำกว่าค่านี้จะถูกตัดทิ้ง ถ้าไม่มีอะไรผ่านเกณฑ์ ระบบจะบอกว่า "
             "\"ไม่พบข้อมูลที่เกี่ยวข้อง\" แทนการยัดเนื้อหาที่ไม่เกี่ยวให้ AI · สูง = เข้มงวด (อาจพลาดบางเรื่อง) · "
             "ต่ำ = หลวม (เสี่ยงได้เนื้อหาไม่เกี่ยว) · สเกลคะแนนต่างกันตามโมเดล embedding ลองปรับตามโมเดลที่ใช้",
    )
    st.session_state.use_mmr = st.toggle(
        "เลือกผลลัพธ์ให้หลากหลาย (MMR)", value=st.session_state.use_mmr,
        help="ป้องกัน top-k เป็น chunk ที่พูดเรื่องเดียวกันซ้ำๆ — เลือกชิ้นที่ทั้งตรงคำถามและไม่ซ้ำกับชิ้นที่เลือกไปแล้ว",
    )
    if st.session_state.use_mmr:
        st.session_state.mmr_lambda = st.slider(
            "สมดุลความตรงคำถาม ↔ ความหลากหลาย (MMR λ)", 0.0, 1.0, st.session_state.mmr_lambda, 0.05,
            help="1.0 = เน้นตรงคำถามล้วน (เหมือนไม่ใช้ MMR) · ต่ำลง = เน้นหลากหลายขึ้น · แนะนำ 0.5–0.8",
        )
    st.caption("ต้อง `ollama pull` โมเดลที่ระบุไว้แล้วในเครื่องก่อนใช้งาน · Graph mode ไม่ต้องใช้ embedding model")

    st.markdown('<div class="section-title">การค้นหาข้อมูล (Graph mode)</div>', unsafe_allow_html=True)
    st.session_state.graph_depth = st.slider(
        "ความลึกในการไล่ความสัมพันธ์ (multi-hop depth)", 1, 5, st.session_state.graph_depth,
        help="เช่น 'ถ้าแก้ A จะกระทบอะไรบ้าง' — 1 = เฉพาะคนที่เรียก A ตรงๆ · 3 = ไล่ต่อไปอีก 2 ทอด "
             "(ผู้เรียกของผู้เรียก ...) · ยิ่งลึกยิ่งครบ แต่ context ที่ส่งให้ AI ยาวขึ้น",
    )
    st.caption("รายการที่มี (?) คือความสัมพันธ์ที่ไม่แน่ใจ (ไม่ทราบชนิดของอ็อบเจ็กต์ที่เรียก จึงเดาจากชื่อเมท็อดอย่างเดียว)")

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
        search_ph = st.empty()
        if st.session_state.use_rag:
            search_ph.markdown(status_html("กำลังค้นข้อมูลจากเอกสาร"), unsafe_allow_html=True)
        context_block, used_sources = "", []
        active_mode = st.session_state.mode if st.session_state.use_rag else None

        no_match = False   # ค้นแล้วไม่มีอะไรผ่านเกณฑ์ (ต่างจาก "ยังไม่มีเอกสาร") — ใช้บอก LLM ให้ตอบตรงๆ ว่าไม่พบ

        if st.session_state.use_rag:
            results = []
            try:
                if st.session_state.mode == "hybrid":
                    res = hybrid_search(
                        vector_store, code_graph, prompt, top_k=st.session_state.top_k,
                        min_score=st.session_state.min_score, use_mmr=st.session_state.use_mmr,
                        mmr_lambda=st.session_state.mmr_lambda, graph_depth=st.session_state.graph_depth,
                        graph_max_nodes=HYBRID_GRAPH_NODES,
                    )
                    for err in res["errors"]:
                        st.warning(f"ค้นหาข้อมูลไม่สำเร็จ ({err}) — ใช้ผลจากอีกแหล่งแทน")
                    built = build_context(res, HYBRID_CONTEXT_CHARS)
                    context_block, used_sources = built["context"], built["sources"]
                    if context_block:
                        note = f"🔀 Hybrid: ใช้ 🕸️ Graph {built['n_graph']} node · 🔎 Vector {built['n_vector']} chunk"
                        if built["truncated"]:
                            note += " · บางส่วนถูกตัดเพื่อไม่ให้ context ยาวเกินไป"
                        st.caption(note)
                    elif vector_store.is_empty() and code_graph.is_empty():
                        st.info("🔀 Hybrid: ยังไม่มีเอกสารใน Knowledge Base "
                                "— เปิดกล่อง 📎 ด้านบนหรือไปที่แท็บ Knowledge Base เพื่ออัปโหลดไฟล์ก่อน")
                    elif not res["errors"]:
                        no_match = True
                        st.info(f"🔀 Hybrid: ไม่พบข้อมูลที่เกี่ยวข้องทั้งจาก Vector (similarity ต่ำกว่า "
                                f"{st.session_state.min_score:.2f}) และ Graph (ไม่พบชื่อฟังก์ชัน/ไฟล์ในคำถาม) "
                                f"— จะไม่ส่ง context ที่ไม่เกี่ยวให้ AI")
                elif st.session_state.mode == "graph":
                    results = code_graph.search(prompt, top_k=st.session_state.top_k,
                                                depth=st.session_state.graph_depth)
                    if not results:
                        st.info("🕸️ Graph mode: ไม่พบฟังก์ชัน/คลาส/โมดูลที่ตรงกับคำถามในกราฟโค้ด "
                                "— ลองระบุชื่อฟังก์ชัน/ไฟล์ให้ตรงกับ dataset หรือสลับไปโหมด Vector")
                else:
                    if vector_store.is_empty():
                        st.info("🔎 Vector mode: ยังไม่มีเอกสารใน Knowledge Base "
                                "— เปิดกล่อง 📎 ด้านบนหรือไปที่แท็บ Knowledge Base เพื่ออัปโหลดไฟล์ก่อน")
                    results = vector_store.search(
                        prompt, top_k=st.session_state.top_k, min_score=st.session_state.min_score,
                        use_mmr=st.session_state.use_mmr, mmr_lambda=st.session_state.mmr_lambda,
                    )
                    if not results and not vector_store.is_empty():
                        no_match = True
                        st.info(f"🔎 Vector mode: ไม่พบเนื้อหาที่เกี่ยวข้อง (similarity ทุก chunk ต่ำกว่า "
                                f"{st.session_state.min_score:.2f}) — จะไม่ส่ง context ที่ไม่เกี่ยวให้ AI "
                                f"ลองถามให้ตรงเอกสารขึ้น หรือลดเกณฑ์ที่ Settings")

                if results:
                    context_block = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
                    used_sources = [r["source"] for r in results]
            except Exception as e:
                st.warning(f"ค้นหาข้อมูลไม่สำเร็จ: {e}")

        system_content = st.session_state.system_prompt
        if context_block:
            if st.session_state.mode == "hybrid":
                system_content += HYBRID_SYSTEM_PROMPT + context_block
            elif st.session_state.mode == "graph":
                system_content += (
                    "\n\nข้อมูลต่อไปนี้คือความสัมพันธ์เชิงโครงสร้างของโค้ด ดึงมาจากกราฟโค้ดจริง ไม่ใช่จากความจำของคุณ "
                    "ข้อมูลแบ่งเป็น 2 หัวข้อที่ทิศทางตรงข้ามกัน ห้ามสับสนหรือสลับกันเด็ดขาด:\n"
                    "- หัวข้อ (A) = สิ่งที่ node นี้เรียกออกไปเอง (this calls out to) — ใช้ตอบคำถามรูปแบบ "
                    "\"X เรียกใช้อะไรบ้าง\"\n"
                    "- หัวข้อ (B) = สิ่งอื่นที่เรียกกลับมาที่ node นี้ (who calls this / impact) — ใช้ตอบคำถามรูปแบบ "
                    "\"ใครเรียกใช้ X บ้าง\" หรือ \"ถ้าแก้ X จะกระทบอะไรบ้าง\"\n"
                    "ตัวอย่าง: ถ้าข้อมูลบอกว่า (A) เรียกใช้: foo, bar และ (B) ใครเรียกใช้: baz "
                    "แล้วผู้ใช้ถาม \"ใครเรียกใช้ฟังก์ชันนี้\" คำตอบที่ถูกคือ baz เท่านั้น (มาจากหัวข้อ B) "
                    "ห้ามตอบ foo หรือ bar เพราะนั่นคือสิ่งที่ฟังก์ชันนี้เรียกออกไปเอง (หัวข้อ A) ไม่ใช่คนเรียกมันเข้ามา\n"
                    "อ้างชื่อไฟล์/ฟังก์ชันให้ตรงกับที่ปรากฏ ห้ามเดาความสัมพันธ์ที่ไม่มีในข้อมูลนี้ "
                    "รายการที่มี (?) คือยังไม่แน่ใจ ให้บอกผู้ใช้ว่าไม่แน่ใจแทนที่จะยืนยัน "
                    "หากข้อมูลไม่พอต่อการตอบให้บอกตามตรงว่าไม่พบในกราฟโค้ดนี้:\n\n" + context_block
                )
            else:
                system_content += (
                    "\n\nข้อมูลต่อไปนี้ดึงมาจากเอกสารที่ผู้ใช้อัปโหลดจริง ให้ใช้เป็นหลักในการตอบหากเกี่ยวข้องกับคำถาม "
                    "ตอบให้ตรงกับเนื้อหาที่ให้มา ห้ามเสริมข้อมูลที่ไม่มีในเนื้อหานี้ขึ้นมาเอง "
                    "หากเนื้อหาที่ให้มาไม่เพียงพอต่อการตอบ ให้บอกตามตรงว่าไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร "
                    "แทนที่จะตอบจากความรู้ทั่วไป:\n\n" + context_block
                )

        elif no_match:
            system_content += (
                "\n\nระบบค้นเอกสารของผู้ใช้แล้วแต่ไม่พบเนื้อหาที่เกี่ยวข้องกับคำถามนี้เลย "
                "หากคำถามเป็นเรื่องเกี่ยวกับเอกสารหรือโค้ดที่ผู้ใช้อัปโหลด ให้บอกตรงๆ ว่าไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร "
                "ห้ามเดาหรือแต่งเนื้อหาของเอกสารขึ้นมาเอง (แนะนำให้ผู้ใช้ถามให้ตรงกับเนื้อหาเอกสารมากขึ้น) "
                "แต่หากเป็นการสนทนาทั่วไปที่ไม่เกี่ยวกับเอกสาร ตอบตามปกติโดยไม่ต้องอ้างเอกสาร"
            )

        search_ph.empty()   # ค้นเสร็จแล้ว เอาแถบสถานะออก
        # ท้าย system prompt: โมเดลขนาดเล็กมักสลับเป็นอังกฤษเมื่อตอบว่า "ไม่พบข้อมูล" จึงย้ำภาษาไว้ท้ายสุด
        system_content += ("\n\nตอบเป็นภาษาเดียวกับที่ผู้ใช้ถามเสมอ (ผู้ใช้ถามเป็นภาษาไทยให้ตอบเป็นภาษาไทย) "
                           "แม้ในกรณีที่ตอบว่าไม่พบข้อมูล")
        history_msgs = build_history(st.session_state.messages)
        ollama_messages = [{"role": "system", "content": system_content}] + history_msgs
        history_chars = sum(len(m["content"]) for m in history_msgs)

        # -- Streaming response พร้อมนับเวลาสด -----------------------------
        # ดึงคำตอบจากโมเดลใน thread แยก แล้วให้ลูปหลักวาดหน้าจอใหม่ทุก UPDATE_INTERVAL วินาที
        # แม้โมเดลจะยังไม่ส่งตัวอักษรแรกมา (ช่วงอ่าน prompt/โหลดโมเดลอาจนานเป็นนาที) — ตัวจับเวลาและแอนิเมชันจะได้ไม่ค้างที่ 0.0s
        answer = ""
        start = time.perf_counter()
        placeholder = st.empty()
        wait_text = (f"ส่งข้อมูลอ้างอิง {len(context_block):,} ตัวอักษรให้โมเดลแล้ว กำลังรอโมเดลเริ่มตอบ"
                     if context_block else "กำลังรอโมเดลเริ่มตอบ")
        placeholder.markdown(bubble_html("assistant", "", elapsed=0.0, thinking=True, mode=active_mode,
                                         wait_text=wait_text), unsafe_allow_html=True)

        if not st.session_state.chat_model:
            answer = "⚠️ ไม่พบโมเดลใน Ollama กรุณา `ollama pull <model>` ก่อนใช้งาน"
        else:
            q: queue.Queue = queue.Queue()
            stop_event = threading.Event()
            chat_model, temperature, num_ctx = (st.session_state.chat_model, st.session_state.temperature,
                                                st.session_state.num_ctx)

            def _pull_stream():
                """รันใน thread แยก: ห้ามเรียก st.* ในนี้ ส่งผลกลับทาง queue เท่านั้น"""
                try:
                    for chunk in client.chat(model=chat_model, messages=ollama_messages, stream=True,
                                             options={"temperature": temperature, "num_ctx": num_ctx}):
                        if stop_event.is_set():
                            return
                        q.put(("chunk", chunk["message"]["content"]))
                    q.put(("done", None))
                except Exception as e:
                    q.put(("error", e))

            threading.Thread(target=_pull_stream, daemon=True).start()
            try:
                finished = False
                while not finished:
                    try:
                        kind, payload = q.get(timeout=UPDATE_INTERVAL)
                        while True:                      # เก็บ chunk ที่ค้างอยู่ให้หมดก่อนวาดหน้าจอ
                            if kind == "chunk":
                                answer += payload
                            elif kind == "done":
                                finished = True
                            else:
                                answer = f"⚠️ เชื่อมต่อ Ollama ไม่สำเร็จ: {payload}"
                                finished = True
                            if finished:
                                break
                            kind, payload = q.get_nowait()
                    except queue.Empty:
                        pass
                    if not finished:
                        placeholder.markdown(
                            bubble_html("assistant", answer, elapsed=time.perf_counter() - start, thinking=True,
                                        mode=active_mode, wait_text=wait_text),
                            unsafe_allow_html=True,
                        )
            finally:
                stop_event.set()    # ถ้าผู้ใช้สั่งรันใหม่/ปิดหน้ากลางคัน ให้ thread เลิกดึงต่อ

        total_elapsed = time.perf_counter() - start
        now = datetime.now().strftime("%H:%M")
        placeholder.markdown(bubble_html("assistant", answer, now, elapsed=total_elapsed, mode=active_mode),
                              unsafe_allow_html=True)
        if used_sources:
            src = ", ".join(sorted(set(used_sources)))
            st.markdown(f'<div class="sources">📎 อ้างอิงจาก: {src}</div>', unsafe_allow_html=True)

        # -- ดูสิ่งที่ส่งให้โมเดลจริง (ไว้ไล่สาเหตุเวลาโมเดลตอบว่า "ไม่พบข้อมูล" ทั้งที่ค้นเจอ) --
        with st.expander("🔍 ดูข้อมูลที่ส่งให้โมเดล", expanded=False):
            total_chars = len(system_content) + history_chars
            st.caption(f"system prompt + ข้อมูลอ้างอิง: {len(system_content):,} ตัวอักษร · ประวัติแชท: {history_chars:,} · "
                       f"รวม {total_chars:,} ตัวอักษร · num_ctx = {st.session_state.num_ctx:,} token")
            if total_chars > st.session_state.num_ctx * 0.9:
                st.warning("ข้อความรวมยาวใกล้/เกิน num_ctx (ภาษาไทยอาจใช้ราว 1 token ต่อ 1 ตัวอักษร) — "
                           "Ollama อาจตัดส่วนต้นของ prompt ทิ้งเงียบๆ ซึ่งเป็นที่อยู่ของข้อมูลอ้างอิง "
                           "ลองเพิ่ม num_ctx หรือลด top-k ที่ Settings")
            st.code(context_block or "(ไม่มีข้อมูลอ้างอิงส่งให้โมเดล)", language=None)

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "ts": now, "elapsed": total_elapsed,
            "sources": used_sources, "mode": active_mode,
        })
