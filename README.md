# PDF Q&A RAG Chatbot 📄🤖

![Status: WIP](https://img.shields.io/badge/status-WIP-orange?style=for-the-badge)

A conversational AI chatbot that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange)

> [!NOTE]
> **Status: Work In Progress (WIP) 🚧**
> This repository is currently under active development. We are actively refining the RAG prompt pipeline, fine-tuning retrieval configurations, and styling the interface.

---

## Features

- Multi-PDF upload and processing
- Semantic search using ChromaDB vector store
- Conversational memory with chat history
- Source attribution per answer
- Powered by Gemini 2.5 Flash + HuggingFace Embeddings

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Gemini 2.5 Flash (Google) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Orchestration | LangChain ConversationalRetrievalChain |
| PDF Parsing | PyPDF2 |

---

## Architecture

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB
                                                           ↓
User Question → Retriever (top-4 chunks) → Gemini 2.5 Flash → Answer + Sources
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Bhushan-git20/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**
```bash
cp .env.example .env
# Edit .env and paste your key
```

Get a free key at [aistudio.google.com](https://aistudio.google.com)

**4. Run the app**
```bash
streamlit run app.py
```

---

## Screenshots

> Coming soon

---

## Author

**Bhushan Damisetti**
[LinkedIn](https://linkedin.com/in/bhushanam-damisetti) · [GitHub](https://github.com/Bhushan-git20)
