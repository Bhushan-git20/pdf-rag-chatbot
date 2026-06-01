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
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .source-box {
        background-color: #f0f4ff;
        border-left: 4px solid #4361ee;
        padding: 10px 15px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    .user-msg {
        background-color: #4361ee;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-msg {
        background-color: #f0f4ff;
        color: #1a1a2e;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 80%;
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
                        vectorstore = process_pdfs(uploaded_files)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
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
                st.markdown(f'<div class="user-msg">🧑 {escaped_content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🤖 {escaped_content}</div>', unsafe_allow_html=True)
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            escaped_source = html.escape(source)
                            st.markdown(f'<div class="source-box"><b>Source {i}:</b> {escaped_source}</div>', unsafe_allow_html=True)

        # Display error if present
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
            st.session_state.error_message = None

        # Input
        user_question = st.chat_input("Ask a question about your PDFs...")
        if user_question:
            try:
                handle_user_input(user_question)
                st.session_state.error_message = None
            except Exception as e:
                st.session_state.error_message = f"❌ Error communicating with AI: {e}"
            st.rerun()


if __name__ == "__main__":
    main()
