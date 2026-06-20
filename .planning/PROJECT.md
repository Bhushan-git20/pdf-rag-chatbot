# PDF RAG Chatbot

## What This Is
A local Streamlit-based web application that allows users to upload PDF documents and converse with them using a Hybrid RAG pipeline (Dense + BM25) reranked by a CrossEncoder and powered by Google Gemini.

## Core Value
Immediate, accurate, and cited answers from user-provided PDF documents without cloud vector database costs, running locally with Streamlit.

## Target Audience
Local users and developers who need to quickly query PDF documents with high retrieval accuracy.

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Streamlit | Fastest way to build a chat UI in Python | Implemented |
| Hybrid Retrieval | Dense embeddings alone fail at keyword searches. BM25 helps. | Implemented |
| CrossEncoder Reranking | Improves retrieval accuracy by scoring BM25 + Dense results | Implemented |
| Gemini 2.5 Flash | Fast, high context window, cheap | Implemented |
| Local ChromaDB | No need for managed cloud DBs for local files | Implemented |

## Requirements

### Validated
- ✓ Upload multiple PDF files
- ✓ Extract and chunk text using RecursiveCharacterTextSplitter
- ✓ Hybrid retrieval (Chroma DB + BM25)
- ✓ Reranking using HuggingFace CrossEncoder
- ✓ Chat UI with message history and sources
- ✓ Thread-safe Gemini embeddings

### Active
- [ ] Implement additional QA enhancements or support other document types

### Out of Scope
- User authentication (local app)
- Persistent database for chat histories (session state only)

## Evolution
This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-06-20 after initialization*
