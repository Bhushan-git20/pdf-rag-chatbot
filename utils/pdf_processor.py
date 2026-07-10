from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.documents import Document
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

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
        chunk_size=512,
        chunk_overlap=128,
        length_function=len
    )
    docs = []
    for item in text_chunks:
        splits = splitter.split_text(item["text"])
        for split in splits:
            # Protect against tiny/empty chunks that crash the CrossEncoder
            if len(split.strip()) > 10:
                docs.append(Document(page_content=split, metadata={"source": item["source"]}))
    return docs


import threading

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def is_transient_embed_error(e):
    error_str = str(e)
    return "429" in error_str or "500" in error_str or "503" in error_str or "RESOURCE_EXHAUSTED" in error_str

class ThreadSafeEmbeddings:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.local = threading.local()

    @property
    def embeddings(self):
        if not hasattr(self.local, "instance"):
            self.local.instance = GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=self.api_key)
        return self.local.instance

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20), retry=retry_if_exception(is_transient_embed_error), reraise=True)
    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20), retry=retry_if_exception(is_transient_embed_error), reraise=True)
    def embed_query(self, text):
        return self.embeddings.embed_query(text)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load HuggingFace embeddings model wrapped for thread safety."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_pdfs(uploaded_files):
    """Full pipeline: extract → split → embed → store in ChromaDB and BM25 → Ensemble."""
    raw_texts = extract_text_from_pdfs(uploaded_files)
    if not raw_texts:
        raise ValueError("No text extracted from PDFs. Check if files are scanned images.")

    lc_docs = split_documents(raw_texts)
    embeddings = load_embeddings()

    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name="pdf_collection"
    )
    
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    
    sparse_retriever = BM25Retriever.from_documents(lc_docs)
    sparse_retriever.k = 8
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]
    )
    
    return ensemble_retriever
