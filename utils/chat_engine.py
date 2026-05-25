import os
import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
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

    response = st.session_state.conversation({
        "question": user_question,
        "chat_history": [
            (msg["content"], st.session_state.chat_history[i + 1]["content"])
            for i, msg in enumerate(st.session_state.chat_history[:-1:2])
        ] if st.session_state.chat_history else []
    })

    answer = response.get("answer", "Sorry, I couldn't find an answer.")
    source_docs = response.get("source_documents", [])

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
