import os
import logging
import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv

load_dotenv()


def get_conversation_chain(vectorstore):
    """Build ConversationalRetrievalChain with Gemini 2.5 Flash."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain


def handle_user_input(user_question: str):
    """Process user question and update chat history."""
    if st.session_state.conversation is None:
        st.error("No conversation chain found. Please process PDFs first.")
        return

    try:
        response = st.session_state.conversation.invoke({
            "question": user_question
        })
    except GoogleAPIError as api_err:
        logging.error(f"Google API Error: {api_err}", exc_info=True)
        raise RuntimeError(f"API Error: {str(api_err)}") from api_err
    except Exception as e:
        logging.error(f"Unexpected error during LLM invocation: {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred: {str(e)}") from e

    answer = response.get("answer", "Sorry, I couldn't find an answer.")
    source_docs = response.get("source_documents", [])

    # Fallback when retriever finds nothing
    if not source_docs:
        answer = "I couldn't find relevant information in the uploaded documents to answer this question. Try rephrasing or check if the topic is covered in your PDFs."

    # Deduplicate sources
    sources = list(dict.fromkeys([
        doc.metadata.get("source", "Unknown source")
        for doc in source_docs
    ]))

    # Update chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
