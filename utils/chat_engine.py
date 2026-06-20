import os
import logging
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    """Load the CrossEncoder model once and cache it."""
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_conversation_chain(retriever):
    """Build LCEL retrieval chain with Gemini 2.5 Flash and Reranker."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )

    # Reranker setup
    cross_encoder = load_cross_encoder()
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # Query expansion prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)

    # QA prompt
    qa_system_prompt = (
        "You are an expert assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question comprehensively. "
        "If you don't know the answer, just say that you don't know. "
        "Provide a detailed, well-structured, and comprehensive answer.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return chain

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def is_transient_error(e):
    if isinstance(e, GoogleAPIError):
        return True
    error_str = str(e)
    return "500 INTERNAL" in error_str or "503" in error_str or "429" in error_str

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_transient_error),
    reraise=True
)
def invoke_with_retry(chain, inputs):
    return chain.invoke(inputs)

def handle_user_input(user_question: str):
    """Process user question and update chat history."""
    if st.session_state.conversation is None:
        st.error("No conversation chain found. Please process PDFs first.")
        return

    # Convert chat history to LangChain message formats 
    # Limit to the last 10 messages (5 interactions) to prevent 400 Context Window Limit
    lc_history = []
    # We slice [-11:-1] because app.py already appended the current user_question
    recent_history = st.session_state.chat_history[-11:-1]
    for msg in recent_history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))

    # (Thread-safety is now handled natively via ThreadSafeEmbeddings in pdf_processor.py)

    try:
        response = invoke_with_retry(st.session_state.conversation, {
            "input": user_question,
            "chat_history": lc_history
        })
    except Exception as e:
        logging.error(f"Failed after retries. Last error: {e}", exc_info=True)
        raise RuntimeError(f"API Error: {str(e)}") from e

    answer = response.get("answer", "Sorry, I couldn't find an answer.")
    source_docs = response.get("context", [])

    # Fallback when retriever finds nothing
    if not source_docs:
        answer = "I couldn't find relevant information in the uploaded documents to answer this question. Try rephrasing or check if the topic is covered in your PDFs."

    # Update chat history
    st.session_state.chat_history.append({
        "role": "bot",
        "content": answer,
        "sources": [{"chunk": doc.page_content, "source": doc.metadata.get("source", "Unknown"), "score": None} for doc in source_docs]
    })
