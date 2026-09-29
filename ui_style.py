import html
import streamlit as st

def inject_css():
    css = r'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    :root{--bg:#0b0d10;--surface:#111419;--surface-2:#171a20;--line:#272c34;--line-soft:#20242b;--text:#f3f4f6;--text-2:#a6adb8;--text-3:#737b88;--success:#8ed6a5;--danger:#ef9a9a}
    html,body,[class*="css"]{font-family:'Inter','Noto Sans Thai',sans-serif!important;background:var(--bg)!important;color:var(--text)!important;-webkit-font-smoothing:antialiased}
    .stApp{background:var(--bg)!important} header[data-testid="stHeader"]{background:transparent!important}
    .block-container{max-width:980px!important;padding:32px 34px 130px!important}
    section[data-testid="stSidebar"]{background:#0e1014!important;border-right:1px solid var(--line-soft)!important}
    section[data-testid="stSidebar"] .block-container{padding:24px 18px!important}
    .brand{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;letter-spacing:-.02em;padding:5px 8px 30px}
    .brand-badge{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:#f4f5f7;color:#111319;font-size:15px}
    .nav-label{color:var(--text-3);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin:26px 8px 12px}
    .sidebar-divider{height:1px;background:var(--line-soft);margin:24px 4px}
    .stButton{margin-bottom:6px!important}.stButton>button{min-height:44px!important;border-radius:9px!important;border:1px solid transparent!important;background:transparent!important;color:var(--text-2)!important;font-size:14px!important;font-weight:500!important;line-height:1.45!important;box-shadow:none!important;transition:background .16s ease,color .16s ease,border-color .16s ease!important}
    .stButton>button:hover{background:var(--surface-2)!important;color:var(--text)!important;transform:none!important;box-shadow:none!important}
    .stButton>button[kind="primary"]{background:var(--surface-2)!important;border-color:var(--line)!important;color:var(--text)!important}
    .hero{padding:10px 0 28px;border-bottom:1px solid var(--line-soft);margin-bottom:24px}
    .hero-badge{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:var(--surface-2);border:1px solid var(--line);margin-bottom:14px;font-size:15px}
    .hero h1{margin:0 0 7px!important;font-size:30px!important;line-height:1.15!important;letter-spacing:-.04em!important;font-weight:700!important;color:var(--text)!important}
    .hero p{margin:0!important;color:var(--text-2)!important;font-size:14px!important;line-height:1.65!important}
    .section-title{color:var(--text)!important;font-size:13px!important;font-weight:600!important;margin:24px 0 10px!important}
    .dime-wrapper{width:100%;margin:0 0 28px;animation:fadeIn .18s ease-out}
    @keyframes fadeIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
    .dime-bubble{width:fit-content;max-width:88%;padding:0;border:0;border-radius:0;background:transparent;color:var(--text);font-size:16px;line-height:1.82;word-break:break-word}
    .dime-bubble-user{margin-left:auto;padding:13px 16px;max-width:78%;background:#1a1e25;border:1px solid #292f38;border-radius:12px}
    .dime-bubble-assistant{max-width:88%}
    .dime-header{display:flex;align-items:center;gap:8px;min-height:22px;margin-bottom:9px}
    .dime-tag{display:inline-flex;align-items:center;gap:6px;padding:0;border:0;background:transparent;font-size:11px;font-weight:600;letter-spacing:.07em;color:var(--text-3)}
    .dime-tag-user{display:none}.dime-tag-ai{color:#8f98a6}
    .dime-pulse-dot{width:6px;height:6px;border-radius:50%;background:#cbd3df;box-shadow:none}
    .dime-pulse-dot.thinking{animation:pulse 1.1s infinite ease-in-out}
    @keyframes pulse{50%{opacity:.35}}
    .dime-content{color:#e8ebef}.dime-content p{margin:0 0 13px}.dime-content p:last-child{margin-bottom:0}
    .dime-content code,code{font-family:'JetBrains Mono',monospace!important;font-size:.88em!important}
    pre{background:#0a0c0f!important;border:1px solid var(--line)!important;border-radius:9px!important;padding:14px!important}
    .dime-meta{display:flex;align-items:center;gap:8px;margin-top:7px;color:var(--text-3);font-size:11px;line-height:1.35}
    .dime-meta-sep{opacity:.45}.dime-mode{padding:3px 7px;border:1px solid var(--line);border-radius:999px;color:var(--text-3);font-size:10px}
    .dime-timer{font-variant-numeric:tabular-nums;font-feature-settings:"tnum";color:#9ba3ae}
    .sources{margin:-11px 0 22px 0;color:var(--text-3);font-size:11px;line-height:1.35.5}
    div[data-testid="stChatInput"]{border:1px solid #303640!important;background:#111419!important;border-radius:14px!important;box-shadow:0 10px 30px rgba(0,0,0,.28)!important;padding:3px 7px!important}
    div[data-testid="stChatInput"]:focus-within{border-color:#555d69!important;box-shadow:0 10px 32px rgba(0,0,0,.35)!important}
    div[data-testid="stChatInput"] textarea{color:var(--text)!important;font-size:16px!important;line-height:1.6!important}
    div[data-testid="stChatInput"] textarea::placeholder{color:#666f7b!important}
    div[data-testid="stChatInput"] button{background:#f1f3f6!important;color:#111318!important;border-radius:9px!important}
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]>div{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:9px!important}
    div[data-baseweb="select"]>div:hover,div[data-baseweb="input"]>div:hover,div[data-baseweb="textarea"]>div:hover{border-color:#3a414c!important}
    label,.stMarkdown,.stCaption{color:var(--text-2)!important}.stSlider [role="slider"]{background:#dce2ea!important}
    div[data-testid="stExpander"]{border:1px solid var(--line-soft)!important;border-radius:10px!important;background:var(--surface)!important}
    .doc-card{display:flex;align-items:center;gap:12px;padding:13px 14px;margin:5px 0;background:var(--surface);border:1px solid var(--line-soft);border-radius:10px}
    .doc-icon{width:32px;height:32px;border-radius:8px;display:grid;place-items:center;background:var(--surface-2);border:1px solid var(--line)}
    .doc-name{color:var(--text);font-size:14px;font-weight:600}.doc-meta{color:var(--text-3);font-size:12px;margin-top:2px}
    .attach-hint{color:var(--text-3);font-size:12px;margin-top:10px}
    .sidebar-stats{color:var(--text-3);font-size:11px;line-height:1.35.9;padding:0 7px}.sidebar-stats b{color:var(--text-2);font-weight:600}
    .status-dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:6px;background:#69717d}.status-dot.ok{background:var(--success)}.status-dot.off{background:var(--danger)}
    #MainMenu,footer{visibility:hidden}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)


def bubble_html(*args, **kwargs) -> str:
    role=kwargs.get("role")
    content=kwargs.get("content",kwargs.get("text",""))
    ts=kwargs.get("ts","")
    elapsed=kwargs.get("elapsed")
    mode=kwargs.get("mode")
    thinking=kwargs.get("thinking",False)

    if args:
        if len(args)>=1: role=str(args[0]).lower()
        if len(args)>=2: content=args[1]
        if len(args)>=3: ts=args[2]
        if len(args)>=4: elapsed=args[3]
        if len(args)>=5: mode=args[4]

    role=role or "assistant"
    is_user=role in ("user","human")

    if elapsed is not None:
        timer=f"{float(elapsed):.1f}s"
        timer_text=f"กำลังคิด · {timer}" if thinking else f"ตอบใน {timer}"
    else:
        timer_text=""

    if is_user:
        header=""
        meta=f'<div class="dime-meta"><span>{html.escape(str(ts))}</span></div>' if ts else ""
        bubble_class="dime-bubble-user"
    else:
        mode_label=""
        if mode=="vector": mode_label='<span class="dime-mode">Vector</span>'
        elif mode=="graph": mode_label='<span class="dime-mode">Graph</span>'
        header=f'''<div class="dime-header"><span class="dime-tag dime-tag-ai"><span class="dime-pulse-dot {"thinking" if thinking else ""}"></span>FLOWCHAT</span></div>'''
        pieces=[]
        if timer_text: pieces.append(f'<span class="dime-timer">{"⏳ " if thinking else ""}{timer_text}</span>')
        if mode_label: pieces.append(mode_label)
        if ts and not thinking: pieces.append(f'<span>{html.escape(str(ts))}</span>')
        sep='<span class="dime-meta-sep">·</span>'
        meta=f'<div class="dime-meta">{sep.join(pieces)}</div>' if pieces else ""
        bubble_class="dime-bubble-assistant"

    safe_content=html.escape(str(content)).replace("\n","<br>")
    if thinking and not safe_content: safe_content='<span style="color:#737b88">กำลังประมวลผลคำตอบ…</span>'

    return f'''<div class="dime-wrapper"><div class="dime-bubble {bubble_class}">{header}<div class="dime-content">{safe_content}</div>{meta}</div></div>'''

apply_custom_ui=inject_css
