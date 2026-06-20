# Architecture Decisions

## Chunk Size: 512 tokens, Overlap: 128
Smaller chunks (512 vs 1000) are retrieved with higher precision when used in combination with a cross-encoder reranker. The 128 token overlap is sufficient to prevent critical context from being cut off mid-sentence while keeping the search index dense and focused.

## Embedding Model: Google text-embedding-004
Swapped from HuggingFace `all-MiniLM-L6-v2` to Google's `text-embedding-004`. It provides superior semantic representation and retrieval accuracy, and is available for free via the Google AI Studio API, eliminating local CPU embedding overhead.

## Vector Store: ChromaDB + BM25 (Hybrid Search)
We use an `EnsembleRetriever` combining ChromaDB (dense semantic search) and BM25 (sparse keyword search). This hybrid approach ensures that both conceptual queries and exact keyword matches (like acronyms or specific IDs) are retrieved reliably.

## Retriever: Hybrid (k=8) + CrossEncoder Reranker (top 3)
The base retrievers pull top 8 chunks (to ensure high recall). These 8 chunks are then scored by a `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`), which outputs the most relevant 3 chunks. This completely eliminates hallucinations caused by irrelevant context while capturing edge cases.

## LLM: Gemini 2.5 Flash
Chosen over GPT-4o for two reasons:
1. Free tier via Google AI Studio — no billing required for demo use.
2. Speed: Flash variant has extremely low latency, making Q&A tasks feel instant.
Quality is excellent when paired with the high-precision Contextual Compression Retriever.

## Memory: LCEL with Chat History
Swapped from the deprecated `ConversationBufferMemory` to the modern LangChain Expression Language (LCEL). We use `create_history_aware_retriever` to inherently rewrite user queries based on the chat history (Query Expansion), significantly improving multi-turn Q&A recall.
