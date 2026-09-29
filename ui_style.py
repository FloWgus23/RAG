"""
ui_style.py — ธีม CSS (โทนสีเดียว indigo, ฟอนต์รองรับไทย-อังกฤษ) และตัว render bubble แชทของ Flowchat

หมายเหตุสำคัญ: ต้องลบบรรทัดว่างออกจาก CSS ก่อนส่งเข้า st.markdown เสมอ (ดู inject_css())
เพราะ Streamlit ใช้ตัวแปลง markdown แบบ CommonMark ซึ่งจะตัดจบ "raw HTML block" ทันทีที่เจอบรรทัดว่าง
แล้วเอาเนื้อหาที่เหลือไปแสดงเป็นข้อความธรรมดาแทนการฝัง <style> จริง
"""

import streamlit as st

from config import MODE_LABELS

_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
    --primary:#5B5BE6;
    --primary-dark:#4746C7;
    --primary-soft:#EEEEFC;
    --primary-grad:linear-gradient(135deg, #5B5BE6 0%, #7A6FF0 100%);
    --graph-accent:#0EA5E9;
    --bg:#F4F5FA;
    --surface:#FFFFFF;
    --surface-alt:#F7F7FC;
    --border:#E7E8F2;
    --text-primary:#181824;
    --text-secondary:#6E6E85;
    --text-faint:#A0A0B5;
    --danger:#E5484D;
    --success:#1FA971;
    --radius-sm:8px; --radius-md:12px; --radius-lg:18px;
    --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px; --space-5:24px; --space-6:32px;
}
html, body, [class*="css"]{ font-family:'IBM Plex Sans Thai','Inter',-apple-system,sans-serif !important; }
h1,h2,h3,h4,.brand,.hero h1{ font-family:'Inter','IBM Plex Sans Thai',sans-serif !important; }
#MainMenu, header, footer { visibility:hidden; }
.block-container{ padding-top:var(--space-4); padding-bottom:var(--space-6); max-width:880px; }
body, .stApp{ background:var(--bg); color:var(--text-primary); }
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .block-container{ padding:var(--space-5) var(--space-3) var(--space-4) var(--space-3); }
.brand{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.15rem; padding:0 var(--space-1) var(--space-5) var(--space-1); color:var(--text-primary); }
.brand-badge{ width:34px; height:34px; border-radius:var(--radius-sm); flex-shrink:0; background:var(--primary-grad); display:flex; align-items:center; justify-content:center; color:white; font-size:17px; box-shadow:0 3px 8px rgba(91,91,230,0.28); }
.nav-label{ font-size:0.68rem; font-weight:700; letter-spacing:0.08em; color:var(--text-faint); text-transform:uppercase; margin:var(--space-4) var(--space-1) var(--space-2) var(--space-1); }
section[data-testid="stSidebar"] .stButton>button{ width:100%; text-align:left; border:1px solid transparent; font-weight:500; font-size:0.9rem; padding:0.5rem 0.75rem; border-radius:var(--radius-sm); margin-bottom:2px; background:transparent; color:var(--text-secondary); transition:all 0.15s ease; }
section[data-testid="stSidebar"] .stButton>button:hover{ background:var(--surface-alt); color:var(--text-primary); }
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{ background:var(--primary-soft) !important; color:var(--primary) !important; font-weight:600; border:1px solid transparent !important; box-shadow:none !important; }
section[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{ background:var(--primary-soft) !important; }
section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stToggle label p{ font-size:0.82rem; color:var(--text-secondary); font-weight:500; }
section[data-testid="stSidebar"] div[data-baseweb="select"]>div{ border-radius:var(--radius-sm); border-color:var(--border); font-size:0.85rem; }
.sidebar-divider{ height:1px; background:var(--border); margin:var(--space-4) 0; }
.sidebar-stats{ font-size:0.76rem; color:var(--text-secondary); line-height:1.9; padding:0 var(--space-1); }
.sidebar-stats b{ color:var(--text-primary); }
.status-dot{ display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:5px; }
.status-dot.ok{ background:var(--success); }
.status-dot.off{ background:var(--danger); }
.hero{ text-align:center; padding:var(--space-5) 0 var(--space-5) 0; }
.hero-badge{ width:48px; height:48px; border-radius:var(--radius-md); margin:0 auto var(--space-3) auto; background:var(--primary-grad); display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 6px 16px rgba(91,91,230,0.25); }
.hero h1{ font-size:1.65rem; font-weight:700; color:var(--text-primary); margin:0 0 var(--space-1) 0; letter-spacing:-0.01em; }
.hero p{ color:var(--text-secondary); font-size:0.88rem; margin:0; }
.chat-row{ display:flex; margin-bottom:var(--space-1); gap:var(--space-3); animation:fadeIn 0.2s ease; }
.chat-row.user{ flex-direction:row-reverse; }
@keyframes fadeIn{ from{opacity:0; transform:translateY(4px);} to{opacity:1; transform:translateY(0);} }
.avatar{ width:30px; height:30px; border-radius:var(--radius-sm); flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:14px; color:white; }
.avatar.assistant{ background:var(--primary-grad); }
.avatar.user{ background:#20202E; }
.bubble-wrap{ display:flex; flex-direction:column; max-width:72%; }
.chat-row.user .bubble-wrap{ align-items:flex-end; }
.bubble{ padding:11px 15px; border-radius:var(--radius-md); line-height:1.65; font-size:0.92rem; white-space:pre-wrap; word-wrap:break-word; }
.bubble.assistant{ background:var(--surface); border:1px solid var(--border); color:var(--text-primary); border-top-left-radius:4px; }
.bubble.user{ background:var(--primary-grad); color:white; border-top-right-radius:4px; }
.meta-row{ display:flex; align-items:center; flex-wrap:wrap; gap:var(--space-2); margin:var(--space-1) 0 0 0; font-size:0.7rem; color:var(--text-faint); }
.time-badge{ display:inline-flex; align-items:center; gap:3px; padding:1px 8px; border-radius:20px; background:var(--surface-alt); border:1px solid var(--border); color:var(--text-secondary); font-size:0.68rem; }
.mode-badge{ display:inline-flex; align-items:center; gap:3px; padding:1px 8px; border-radius:20px; background:var(--primary-soft); color:var(--primary); font-size:0.68rem; font-weight:600; }
.mode-badge.graph{ background:#E6F7F3; color:#0F8A6B; }
.sources{ font-size:0.74rem; color:var(--text-faint); margin:2px 0 var(--space-4) 42px; }
.thinking-dots span{ display:inline-block; width:5px; height:5px; margin-right:3px; border-radius:50%; background:var(--primary); animation:bounce 1.1s infinite ease-in-out; }
.thinking-dots span:nth-child(2){ animation-delay:0.15s; }
.thinking-dots span:nth-child(3){ animation-delay:0.3s; }
@keyframes bounce{ 0%,80%,100%{transform:scale(0.6); opacity:0.4;} 40%{transform:scale(1); opacity:1;} }
div[data-testid="stExpander"]{ border:1px solid var(--border) !important; border-radius:var(--radius-md) !important; background:var(--surface); margin-bottom:var(--space-4); overflow:hidden; }
div[data-testid="stExpander"] summary{ font-size:0.85rem; font-weight:500; color:var(--text-secondary); padding:var(--space-3) var(--space-4) !important; }
.attach-hint{ font-size:0.76rem; color:var(--text-secondary); margin-top:var(--space-2); padding:var(--space-2) var(--space-3); background:var(--surface-alt); border-radius:var(--radius-sm); }
.doc-card{ display:flex; align-items:center; gap:var(--space-3); background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-md); padding:var(--space-3) var(--space-4); margin-bottom:var(--space-2); }
.doc-card .doc-icon{ font-size:1.3rem; }
.doc-card .doc-name{ font-weight:600; color:var(--text-primary); font-size:0.88rem; }
.doc-card .doc-meta{ font-size:0.74rem; color:var(--text-faint); }
.section-title{ font-size:0.95rem; font-weight:700; color:var(--text-primary); margin:var(--space-4) 0 var(--space-3) 0; }
[data-testid="stChatInput"]{ border-color:var(--border); }
</style>
"""


def inject_css():
    """ส่ง CSS เข้า Streamlit — ลบบรรทัดว่างก่อนเสมอ กันบั๊ก raw-HTML block ถูกตัดจบก่อนเวลา"""
    st.markdown("\n".join(line for line in _CSS.splitlines() if line.strip()), unsafe_allow_html=True)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} วินาที"
    m, s = divmod(seconds, 60)
    return f"{int(m)} นาที {s:.0f} วินาที"


def bubble_html(role: str, content: str, ts: str = "", elapsed: float | None = None,
                 thinking: bool = False, mode: str | None = None) -> str:
    avatar = "🙂" if role == "user" else "💬"
    meta = f'<span>{ts}</span>' if ts else ""
    if mode and role == "assistant":
        icon, label = MODE_LABELS.get(mode, ("", mode))
        badge_class = "mode-badge graph" if mode == "graph" else "mode-badge"
        meta += f'<span class="{badge_class}">{icon} {label}</span>'
    if thinking:
        meta += f'<span class="time-badge">⏳ {elapsed:.1f}s</span>' if elapsed else '<span class="time-badge">⏳</span>'
    elif elapsed is not None:
        meta += f'<span class="time-badge">⚡ {format_elapsed(elapsed)}</span>'
    body = content if content else '<span class="thinking-dots"><span></span><span></span><span></span></span> กำลังคิด...'
    return f"""
    <div class="chat-row {role}">
        <div class="avatar {role}">{avatar}</div>
        <div class="bubble-wrap">
            <div class="bubble {role}">{body}</div>
            <div class="meta-row">{meta}</div>
        </div>
    </div>
    """
