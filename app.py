"""
Arabic RAG Chatbot — Streamlit Web Interface
Run: streamlit run app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.rag_engine import ArabicRAGChatbot
from src.document_loader import load_document_from_bytes

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="عربي RAG | Arabic Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --accent-primary: #238636;
    --accent-secondary: #1f6feb;
    --accent-glow: rgba(35,134,54,0.15);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --border: #30363d;
    --user-bubble: #1f6feb20;
    --bot-bubble: #23863620;
    --danger: #da3633;
    --warning: #d29922;
  }

  .stApp { background: var(--bg-primary); }
  .stApp > header { background: transparent; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
  }

  /* Typography */
  * { font-family: 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif !important; }
  h1, h2, h3 { color: var(--text-primary) !important; }

  /* Chat messages */
  .message-container { margin: 12px 0; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

  .user-message {
    background: var(--user-bubble);
    border: 1px solid rgba(31,111,235,0.3);
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0 8px 15%;
    color: var(--text-primary);
    text-align: right;
    direction: rtl;
    font-size: 0.95rem;
    line-height: 1.7;
  }

  .bot-message {
    background: var(--bot-bubble);
    border: 1px solid rgba(35,134,54,0.3);
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 15% 8px 0;
    color: var(--text-primary);
    text-align: right;
    direction: rtl;
    font-size: 0.95rem;
    line-height: 1.8;
  }

  .source-badge {
    display: inline-block;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 6px 3px 0;
    font-family: 'JetBrains Mono', monospace !important;
    direction: ltr;
  }

  .meta-line {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace !important;
    direction: ltr;
  }

  /* Stats cards */
  .stat-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
  }
  .stat-number { font-size: 1.8rem; font-weight: 600; color: var(--accent-primary); }
  .stat-label { font-size: 0.75rem; color: var(--text-secondary); }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }

  /* Input */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    direction: rtl;
  }

  .stButton > button {
    background: var(--accent-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--accent-glow);
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: var(--bg-tertiary);
    border: 1px dashed var(--border);
    border-radius: 8px;
  }

  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
    font-size: 0.9rem;
  }
  .empty-icon { font-size: 3rem; margin-bottom: 12px; }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        'chatbot': None,
        'messages': [],
        'docs_loaded': 0,
        'total_chunks': 0,
        'total_tokens': 0,
        'api_key': os.getenv('GROQ_API_KEY', ''),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")
    st.markdown("---")

    # API Key
    api_key = st.text_input(
        "🔑 Groq API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="gsk_...",
        help="احصل على مفتاح مجاني من console.groq.com"
    )
    st.session_state.api_key = api_key

    # Model selector
    model = st.selectbox(
        "🤖 النموذج",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        help="llama-3.3-70b أدق، llama-3.1-8b أسرع"
    )

    top_k = st.slider("📎 عدد المقاطع المسترجعة", 2, 8, 4,
                      help="عدد أكبر = إجابة أشمل، لكن أبطأ")

    st.markdown("---")
    st.markdown("### 📁 رفع المستندات")

    uploaded_files = st.file_uploader(
        "ارفع ملفات PDF أو TXT",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("🚀 فهرسة المستندات", use_container_width=True):
        if not api_key:
            st.error("⚠️ أدخل Groq API Key أولاً")
        else:
            with st.spinner("جاري معالجة المستندات..."):
                try:
                    chatbot = ArabicRAGChatbot(api_key=api_key, model=model)

                    docs = []
                    progress = st.progress(0)
                    for i, f in enumerate(uploaded_files):
                        content, name = load_document_from_bytes(f.read(), f.name)
                        docs.append((content, name))
                        progress.progress((i + 1) / len(uploaded_files))

                    chatbot.add_documents(docs)
                    st.session_state.chatbot = chatbot
                    st.session_state.docs_loaded = len(docs)
                    st.session_state.total_chunks = len(chatbot.retriever.chunks)
                    st.session_state.messages = []
                    progress.empty()
                    st.success(f"✅ تم فهرسة {len(docs)} مستند بنجاح!")
                except Exception as e:
                    st.error(f"❌ خطأ: {e}")

    # Stats
    if st.session_state.docs_loaded > 0:
        st.markdown("---")
        st.markdown("### 📊 إحصائيات")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number">{st.session_state.docs_loaded}</div>
                <div class="stat-label">مستندات</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number">{st.session_state.total_chunks}</div>
                <div class="stat-label">مقاطع</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_history()
        st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align:center; color:#484f58; font-size:0.72rem; margin-top:20px">
    Arabic RAG Chatbot<br>
    Powered by Groq + LLaMA 3.3<br>
    <a href="https://github.com/Shog7777/arabic-rag-chatbot" style="color:#238636">github.com/Shog7777</a>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Area ────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
  <div style="font-size:2.5rem">🧠</div>
  <div>
    <h1 style="margin:0; font-size:1.5rem; color:#e6edf3">Arabic RAG Chatbot</h1>
    <p style="margin:4px 0 0; color:#8b949e; font-size:0.85rem; direction:rtl">
      اسأل أي سؤال عن مستنداتك — الإجابات مستخرجة مباشرة من محتواك
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


# Chat display
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        if st.session_state.docs_loaded == 0:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <div><strong>ارفع مستنداتك للبدء</strong></div>
              <div style="margin-top:8px">يدعم ملفات PDF وTXT وMarkdown باللغة العربية والإنجليزية</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">💬</div>
              <div><strong>المستندات جاهزة — ابدأ بالسؤال</strong></div>
            </div>
            """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="message-container">
              <div class="user-message">{msg['content']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            sources_html = ''.join(
                f'<span class="source-badge">📄 {s}</span>'
                for s in msg.get('sources', [])
            )
            meta = f"chunks: {msg.get('chunks_used',0)} | tokens: {msg.get('tokens_used',0)}"
            st.markdown(f"""
            <div class="message-container">
              <div class="bot-message">
                {msg['content']}
                {f'<div style="margin-top:10px;text-align:right">{sources_html}</div>' if sources_html else ''}
                <div class="meta-line">{meta}</div>
              </div>
            </div>""", unsafe_allow_html=True)


# Input
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    question = st.text_input(
        "سؤالك",
        placeholder="اكتب سؤالك هنا... مثال: ما هي أهداف رؤية 2030؟",
        label_visibility="collapsed",
        key="question_input"
    )

with col2:
    send_clicked = st.button("إرسال ➤", use_container_width=True)


# Suggested questions
if st.session_state.docs_loaded > 0 and not st.session_state.messages:
    st.markdown("""
    <p style="color:#8b949e; font-size:0.8rem; margin-top:12px; text-align:right">
    💡 أمثلة على الأسئلة:
    </p>""", unsafe_allow_html=True)

    q_cols = st.columns(3)
    suggestions = [
        "ما هي النقاط الرئيسية في هذا المستند؟",
        "لخّص المحتوى في ثلاث نقاط",
        "ما التوصيات الواردة في الوثيقة؟"
    ]
    for col, q in zip(q_cols, suggestions):
        with col:
            if st.button(q, use_container_width=True, key=f"sugg_{q[:10]}"):
                question = q
                send_clicked = True


# Handle send
if (send_clicked or question) and question.strip():
    if not st.session_state.chatbot:
        st.error("⚠️ ارفع مستنداتك أولاً من الشريط الجانبي")
    else:
        # Update model if changed
        st.session_state.chatbot.model = model

        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': question.strip()})

        with st.spinner("🔍 جاري البحث والإجابة..."):
            try:
                result = st.session_state.chatbot.ask(question.strip(), top_k=top_k)
                st.session_state.total_tokens += result.get('tokens_used', 0)
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources'],
                    'chunks_used': result['chunks_used'],
                    'tokens_used': result.get('tokens_used', 0),
                })
            except Exception as e:
                st.error(f"❌ خطأ في الإجابة: {e}")

        st.rerun()
