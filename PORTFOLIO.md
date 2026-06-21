# Portfolio Links

This file tracks the live deployments and important links for portfolio updates.

## PDF Q&A RAG Chatbot

- **Live Demo**: [https://huggingface.co/spaces/Bhushanam/pdf-rag-chatbot](https://huggingface.co/spaces/Bhushanam/pdf-rag-chatbot)
- **Source Code**: [https://github.com/Bhushan-git20/pdf-rag-chatbot](https://github.com/Bhushan-git20/pdf-rag-chatbot)
- **Key Highlights for Resume/Portfolio**:
  - Engineered an end-to-end Retrieval-Augmented Generation (RAG) pipeline utilizing **LangChain (LCEL)**, **Streamlit**, and **Google Gemini 2.5 Flash**.
  - Implemented an advanced hybrid search retrieval system combining sparse (BM25) and dense (ChromaDB) search, further refined by a **CrossEncoder reranker** (`ms-marco-MiniLM-L-6-v2`) to eliminate hallucinations.
  - Developed a custom, thread-safe Google Embeddings implementation with robust exponential backoff (`tenacity`) to gracefully handle high concurrency and rate limits in cloud deployments.
  - Achieved full source attribution with real-time confidence scores and contextual conversational memory.
