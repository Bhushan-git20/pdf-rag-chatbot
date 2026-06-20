import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import html
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from utils.pdf_processor import process_pdfs
from utils.chat_engine import get_conversation_chain, handle_user_input

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    :root {
        /* LIGHT MODE DEFAULT (Option A: Gemini Light) */
        --bg-color: #FFFFFF;
        --text-color: #1F2937;
        --sidebar-bg: rgba(243, 244, 246, 0.7);
        --sidebar-border: rgba(0, 0, 0, 0.05);
        --header-gradient: -webkit-linear-gradient(45deg, #1A73E8, #8AB4F8);
        --user-bubble-bg: linear-gradient(135deg, #1A73E8 0%, #8AB4F8 100%);
        --user-text: #FFFFFF;
        --user-bubble-shadow: 0 4px 15px rgba(26, 115, 232, 0.2);
        --bot-bubble-bg: #FFFFFF;
        --bot-bubble-border: rgba(0, 0, 0, 0.08);
        --bot-bubble-shadow: 0 4px 6px rgba(0,0,0,0.02);
        --source-bg: #F9FAFB;
        --source-border: #6366F1;
        --source-text: #4B5563;
        --input-bg: #FFFFFF;
        --input-border: rgba(0, 0, 0, 0.1);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            /* DARK MODE (Option B: Glowing Glassmorphic) */
            --bg-color: #0f111a;
            --text-color: #e2e8f0;
            --sidebar-bg: rgba(20, 24, 36, 0.7);
            --sidebar-border: rgba(255, 255, 255, 0.05);
            --header-gradient: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
            --user-bubble-bg: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
            --user-text: #ffffff;
            --user-bubble-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
            --bot-bubble-bg: rgba(255, 255, 255, 0.05);
            --bot-bubble-border: rgba(255, 255, 255, 0.1);
            --bot-bubble-shadow: 0 4px 6px rgba(0,0,0,0.1);
            --source-bg: rgba(20, 24, 36, 0.8);
            --source-border: #8B5CF6; /* AI Purple neon */
            --source-text: #a0aec0;
            --input-bg: rgba(20, 24, 36, 0.9);
            --input-border: rgba(255, 255, 255, 0.1);
        }
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background and global theme overrides */
    .stApp {
        background: var(--bg-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Hide Streamlit header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--sidebar-border);
    }
    
    /* Main Header */
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--header-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    /* Chat bubbles */
    .user-msg {
        background: var(--user-bubble-bg);
        color: var(--user-text);
        padding: 12px 18px;
        border-radius: 24px 24px 4px 24px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: var(--user-bubble-shadow);
        font-weight: 400;
    }
    
    .bot-msg {
        background-color: var(--bot-bubble-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--bot-bubble-border);
        color: var(--text-color);
        padding: 15px 20px;
        border-radius: 24px 24px 24px 4px;
        margin: 8px 0;
        max-width: 85%;
        box-shadow: var(--bot-bubble-shadow);
        line-height: 1.6;
    }
    
    /* Source boxes */
    .source-box {
        background-color: var(--source-bg);
        border-left: 3px solid var(--source-border);
        padding: 10px 15px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-top: 10px;
        color: var(--source-text);
        transition: all 0.2s ease;
    }
    .source-box:hover {
        opacity: 0.9;
        transform: translateX(2px);
    }
    
    /* Inputs */
    .stChatInputContainer {
        border: 1px solid var(--input-border) !important;
        border-radius: 24px !important;
        background: var(--input-bg) !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
    }

    /* Fix sidebar text contrast */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] span {
        color: #cbd5e0 !important;
    }

    /* Fix file uploader helper text */
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #a0aec0 !important;
    }

    /* Fix all general paragraph text */
    .stApp p, .stApp span, .stApp div {
        color: #e2e8f0;
    }

    /* Fix info box text */
    [data-testid="stAlert"] p {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">📄 PDF Q&A Chatbot</p>', unsafe_allow_html=True)
    st.markdown("Upload PDFs, ask questions, get answers with sources.")

    # Session state init
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "error_message" not in st.session_state:
        st.session_state.error_message = None

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    try:
                        retriever = process_pdfs(uploaded_files)
                        st.session_state.conversation = get_conversation_chain(retriever)
                        st.session_state.processed = True
                        st.session_state.chat_history = []
                        st.success(f"✅ {len(uploaded_files)} PDF(s) processed!")
                    except ValueError as e:
                        st.error(f"❌ Could not process PDFs: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
            else:
                st.warning("Please upload at least one PDF.")

        if st.session_state.processed:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        st.divider()
        st.markdown("**Stack:** LangChain · ChromaDB · Gemini 2.5 Flash · Streamlit")

    # Chat area
    if not st.session_state.processed:
        st.info("👈 Upload PDFs from the sidebar to get started.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            escaped_content = html.escape(message["content"])
            if message["role"] == "user":
                st.markdown(f'<div class="user-msg">{escaped_content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{escaped_content}</div>', unsafe_allow_html=True)
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for i, src in enumerate(message["sources"], 1):
                            source_text = f"<b>Source {i}:</b> {html.escape(src['source'])}<br>"
                            source_text += f"<i>Score: {src['score']:.2f}</i><br>" if src['score'] is not None else ""
                            source_text += f"<blockquote>{html.escape(src['chunk'])}</blockquote>"
                            st.markdown(f'<div class="source-box">{source_text}</div>', unsafe_allow_html=True)

        # Display error if present
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            st.session_state.error_message = None

        # Input
        user_question = st.chat_input("Ask a question about your PDFs...")
        if user_question:
            # Immediately show the user's message
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            escaped_q = html.escape(user_question)
            st.markdown(f'<div class="user-msg">{escaped_q}</div>', unsafe_allow_html=True)
            
            with st.spinner("Thinking..."):
                try:
                    handle_user_input(user_question)
                    st.session_state.error_message = None
                except Exception as e:
                    st.session_state.error_message = f"❌ Error communicating with AI: {e}"
                    # Remove the failed question from history so they can retry
                    st.session_state.chat_history.pop()
            st.rerun()


if __name__ == "__main__":
    main()
