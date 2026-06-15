import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.rag_engine import GeminiRAGChatbot
from src.document_loader import load_from_bytes

st.set_page_config(
    page_title="مساعد المستندات الذكي",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');
* { font-family: 'IBM Plex Sans Arabic', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%); }
[data-testid="stSidebar"] { background: rgba(13,17,23,0.95) !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
.hero { background: linear-gradient(135deg, rgba(66,133,244,0.15) 0%, rgba(52,168,83,0.1) 50%, rgba(251,188,4,0.08) 100%); border: 1px solid rgba(66,133,244,0.2); border-radius: 20px; padding: 28px 36px; margin-bottom: 28px; }
.hero-title { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #4285f4, #34a853, #fbbc04); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; direction: rtl; }
.hero-sub { color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 6px; direction: rtl; }
.msg-user { background: linear-gradient(135deg, rgba(66,133,244,0.12), rgba(66,133,244,0.06)); border: 1px solid rgba(66,133,244,0.25); border-radius: 18px 18px 6px 18px; padding: 16px 20px; margin: 10px 0 10px 12%; color: #e8eaed; direction: rtl; text-align: right; font-size: 0.95rem; line-height: 1.8; }
.msg-bot { background: linear-gradient(135deg, rgba(52,168,83,0.08), rgba(52,168,83,0.03)); border: 1px solid rgba(52,168,83,0.2); border-radius: 18px 18px 18px 6px; padding: 16px 20px; margin: 10px 12% 10px 0; color: #e8eaed; direction: rtl; text-align: right; font-size: 0.95rem; line-height: 1.9; }
.src-badge { display: inline-flex; align-items: center; gap: 4px; background: rgba(251,188,4,0.1); border: 1px solid rgba(251,188,4,0.25); color: #fbbc04; font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; margin: 6px 3px 0; }
.meta-info { color: rgba(255,255,255,0.25); font-size: 0.68rem; margin-top: 10px; direction: ltr; text-align: left; }
.stat { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px; text-align: center; }
.stat-n { font-size: 2rem; font-weight: 700; color: #4285f4; line-height: 1; }
.stat-l { font-size: 0.72rem; color: rgba(255,255,255,0.4); margin-top: 4px; }
.empty { text-align: center; padding: 80px 20px; color: rgba(255,255,255,0.2); }
.empty-icon { font-size: 4rem; margin-bottom: 16px; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.4); }
.gemini-badge { display: inline-flex; align-items: center; gap: 6px; background: linear-gradient(135deg, rgba(66,133,244,0.15), rgba(234,67,53,0.1)); border: 1px solid rgba(66,133,244,0.3); border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; color: #4285f4; margin-bottom: 16px; }
.fmt-tag { display: inline-block; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 2px 8px; font-size: 0.7rem; color: rgba(255,255,255,0.5); margin: 2px; }
.stButton > button { background: linear-gradient(135deg, #4285f4, #1a73e8) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
.stTextInput > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; color: #e8eaed !important; direction: rtl; }
hr { border-color: rgba(255,255,255,0.06) !important; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in {'bot': None, 'msgs': [], 'api_key': '', 'docs_loaded': 0, 'total_chunks': 0, 'indexed': False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="gemini-badge">✨ Powered by Gemini 2.0 Flash</div>', unsafe_allow_html=True)

    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("", value=st.session_state.api_key,
                             type="password", placeholder="AIza...",
                             label_visibility="collapsed")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.bot = None
        st.session_state.indexed = False

    if st.session_state.api_key:
        st.success("✅ API Key محفوظ")

    st.markdown("---")
    st.markdown("### 📂 رفع المستندات")
    st.markdown('<div style="margin-bottom:10px"><span class="fmt-tag">📄 PDF</span><span class="fmt-tag">📝 Word</span><span class="fmt-tag">📊 Excel</span><span class="fmt-tag">📃 TXT</span><span class="fmt-tag">📋 MD</span></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=['pdf','docx','xlsx','xls','txt','md','doc'],
                                  accept_multiple_files=True, label_visibility="collapsed")

    if st.button("🚀 تحليل وفهرسة المستندات", use_container_width=True):
        if not st.session_state.api_key:
            st.error("⚠️ أدخل Gemini API Key أولاً")
        elif not uploaded:
            st.error("⚠️ ارفع ملفاً أولاً")
        else:
            docs = []
            errors = []
            prog = st.progress(0)
            status = st.empty()

            for i, f in enumerate(uploaded):
                status.text(f"⏳ معالجة {f.name}...")
                try:
                    content, name = load_from_bytes(f.read(), f.name)
                    if len(content) > 50:
                        docs.append((content, name))
                    else:
                        errors.append(f"⚠️ {f.name}: فارغ")
                except Exception as e:
                    errors.append(f"❌ {f.name}: {e}")
                prog.progress((i+1)/len(uploaded))

            prog.empty()
            status.empty()

            if docs:
                try:
                    bot = GeminiRAGChatbot(api_key=st.session_state.api_key)
                    bot.add_documents(docs)
                    st.session_state.bot = bot
                    st.session_state.docs_loaded = bot.docs_loaded
                    st.session_state.total_chunks = bot.total_chunks
                    st.session_state.indexed = True
                    st.session_state.msgs = []
                    st.success(f"✅ تم فهرسة {len(docs)} ملف | {bot.total_chunks} مقطع")
                except Exception as e:
                    st.error(f"❌ خطأ في Gemini: {e}")
            for err in errors:
                st.warning(err)

    if st.session_state.docs_loaded > 0:
        st.markdown("---")
        st.markdown("### 📊 إحصائيات")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat"><div class="stat-n">{st.session_state.docs_loaded}</div><div class="stat-l">ملفات</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat"><div class="stat-n">{st.session_state.total_chunks}</div><div class="stat-l">مقطع</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.msgs = []
            st.session_state.bot = None
            st.session_state.docs_loaded = 0
            st.session_state.total_chunks = 0
            st.session_state.indexed = False
            st.rerun()
    with c2:
        top_k = st.selectbox("المقاطع", [3,5,8,10], index=1, label_visibility="collapsed")

    st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.15);font-size:0.7rem;margin-top:20px">Arabic RAG v2 · <a href="https://github.com/Shog7777/arabic-rag-chatbot" style="color:#4285f4">@Shog7777</a></div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="display:flex;align-items:center;gap:16px">
        <div style="font-size:2.8rem">✨</div>
        <div>
            <div class="hero-title">مساعد المستندات الذكي</div>
            <div class="hero-sub">ارفع ملفاتك واسأل بالعربية — مدعوم بـ Gemini 2.0 Flash</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat messages
if not st.session_state.msgs:
    if not st.session_state.indexed:
        st.markdown('<div class="empty"><div class="empty-icon">📂</div><div class="empty-title">ارفع مستنداتك للبدء</div><div style="font-size:0.82rem;margin-top:8px">PDF · Word · Excel · TXT</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty"><div class="empty-icon">💬</div><div class="empty-title">المستندات جاهزة — اسأل أي سؤال</div></div>', unsafe_allow_html=True)

for msg in st.session_state.msgs:
    if msg['role'] == 'user':
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        srcs = ''.join(f'<span class="src-badge">📄 {s}</span>' for s in msg.get('sources', []))
        st.markdown(f"""<div class="msg-bot">
            {msg['content']}
            {f'<div style="margin-top:12px;text-align:right">{srcs}</div>' if srcs else ''}
            <div class="meta-info">chunks: {msg.get('chunks_used',0)} · gemini-2.0-flash</div>
        </div>""", unsafe_allow_html=True)

# Suggestions
if st.session_state.indexed and not st.session_state.msgs:
    st.markdown('<p style="color:rgba(255,255,255,0.3);font-size:0.8rem;text-align:right;margin-top:16px">💡 جرب:</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, q in zip(cols, ["لخّص المحتوى في نقاط", "ما أهم المعلومات؟", "ما التوصيات الواردة؟"]):
        with col:
            if st.button(q, use_container_width=True, key=f"s_{q[:6]}"):
                st.session_state._pq = q
                st.rerun()

# Input
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns([5,1])
with c1:
    question = st.text_input("", placeholder="اكتب سؤالك هنا...", label_visibility="collapsed", key="q_in")
with c2:
    send = st.button("إرسال ✦", use_container_width=True)

if hasattr(st.session_state, '_pq'):
    question = st.session_state._pq
    del st.session_state._pq
    send = True

if send and question and question.strip():
    if not st.session_state.bot or not st.session_state.indexed:
        st.error("⚠️ ارفع مستنداتك وافهرسها أولاً")
    else:
        st.session_state.msgs.append({'role': 'user', 'content': question.strip()})
        with st.spinner("✨ Gemini يفكر..."):
            try:
                result = st.session_state.bot.ask(question.strip(), top_k=top_k)
                st.session_state.msgs.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources'],
                    'chunks_used': result['chunks_used'],
                })
            except Exception as e:
                st.error(f"❌ {e}")
        st.rerun()
