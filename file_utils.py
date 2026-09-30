"""
file_utils.py — ฟังก์ชันช่วยจัดการไฟล์ที่อัปโหลด ใช้ร่วมกันทั้งหน้า Knowledge Base และช่องแนบไฟล์ในหน้าแชท
ไฟล์ .py จะถูกเพิ่มเข้าทั้ง VectorStore (Vector mode) และ CodeGraph (Graph mode) พร้อมกัน
ไฟล์ประเภทอื่น (.pdf/.docx/.txt/.md) เข้าแค่ VectorStore
"""

import time

import streamlit as st

from config import FILE_ICONS
from vector_rag import VectorStore, read_file
from graph_rag import CodeGraph


def file_icon(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return FILE_ICONS.get(ext, "📄")


def process_uploaded_files(files, vector_store: VectorStore, code_graph: CodeGraph):
    """
    รับไฟล์จาก st.file_uploader มา index เข้า Vector store และ (ถ้าเป็น .py) Code graph ด้วย

    ไฟล์ .py ทั้งหมดในรอบเดียวกันจะถูก add_file(auto_finalize=False) ก่อน แล้ว finalize()
    กราฟครั้งเดียวหลังลูปจบ — เร็วกว่าและถูกต้องกว่าการ finalize ทุกครั้งที่เพิ่มทีละไฟล์
    (ความสัมพันธ์ข้ามไฟล์ เช่น A import B จะครบก็ต่อเมื่อไฟล์ทั้งชุดถูกเพิ่มหมดแล้ว)
    """
    if not files:
        return
    existing = set(vector_store.document_names())
    graph_touched = False

    for f in files:
        if f.name in existing:
            continue
        t0 = time.perf_counter()
        with st.spinner(f"กำลังประมวลผล {f.name} ..."):
            text = read_file(f)
            n_chunks = vector_store.add_document(f.name, text)
            n_nodes = 0
            if f.name.endswith(".py"):
                n_nodes = code_graph.add_file(f.name, text, auto_finalize=False)
                graph_touched = True

        msg = f"เพิ่ม {f.name} แล้ว ({n_chunks} chunks"
        if n_nodes:
            msg += f", {n_nodes} code nodes"
        msg += f", {time.perf_counter() - t0:.1f}s)"
        st.toast(msg, icon="✅")

    if graph_touched:
        code_graph.finalize()  # คำนวณความสัมพันธ์ข้ามไฟล์ครั้งเดียวหลังอัปโหลดครบทุกไฟล์


def remove_document_everywhere(filename: str, vector_store: VectorStore, code_graph: CodeGraph):
    vector_store.remove_document(filename)
    if filename.endswith(".py"):
        code_graph.remove_file(filename)  # ลบแล้ว finalize ใหม่ในตัวอยู่แล้ว (ดู graph_rag.py)
