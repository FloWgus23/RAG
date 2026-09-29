"""
ollama_client.py — จัดการการเชื่อมต่อกับ Ollama
ใช้ st.cache_resource เก็บ Client ตัวเดียวไว้ใช้ตลอด session (แทนเปิด connection ใหม่ทุกครั้ง)
และ st.cache_data (ttl สั้นๆ) เก็บรายชื่อโมเดล กัน ollama.list() ถูกยิงซ้ำทุก rerun ของ Streamlit
"""

import streamlit as st
import ollama

from config import EMBED_HINTS


@st.cache_resource
def get_client() -> ollama.Client:
    return ollama.Client()


@st.cache_data(ttl=15)
def get_ollama_models(_client: ollama.Client) -> list[str]:
    """ขีดเส้นใต้ _client: นำหน้าด้วย _ บอก Streamlit ว่าไม่ต้อง hash ตัวแปรนี้ตอน cache"""
    try:
        resp = _client.list()
        return [m["model"] for m in resp.get("models", [])]
    except Exception:
        return []


def is_embedding_model(name: str) -> bool:
    n = name.lower()
    return any(hint in n for hint in EMBED_HINTS)


def chat_models_only(all_models: list[str]) -> list[str]:
    """ตัดโมเดลประเภท embedding ออก เหลือแต่โมเดลที่ใช้แชทได้จริง"""
    filtered = [m for m in all_models if not is_embedding_model(m)]
    return filtered or all_models
