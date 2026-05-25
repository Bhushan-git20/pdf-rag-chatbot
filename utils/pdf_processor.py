from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


def extract_text_from_pdfs(uploaded_files):
    """Extract raw text from uploaded PDF files."""
    all_text = []
    for pdf_file in uploaded_files:
        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_text.append({
                    "text": text,
                    "source": f"{pdf_file.name} (Page {page_num + 1})"
                })
    return all_text


def split_documents(text_chunks):
    """Split text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = []
    metadatas = []
    for item in text_chunks:
        splits = splitter.split_text(item["text"])
        for split in splits:
            docs.append(split)
            metadatas.append({"source": item["source"]})
    return docs, metadatas


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load HuggingFace embeddings model (cached)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def process_pdfs(uploaded_files):
    """Full pipeline: extract → split → embed → store in ChromaDB."""
    raw_texts = extract_text_from_pdfs(uploaded_files)
    if not raw_texts:
        raise ValueError("No text extracted from PDFs. Check if files are scanned images.")

    docs, metadatas = split_documents(raw_texts)
    embeddings = load_embeddings()

    vectorstore = Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="pdf_collection"
    )
    return vectorstore
