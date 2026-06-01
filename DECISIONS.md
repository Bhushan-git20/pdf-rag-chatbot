# Architecture Decisions

## Chunk Size: 1000 tokens, Overlap: 200
Balances context retention vs retrieval precision.
Larger chunks retain more context but slow down retrieval and increase noise.
Smaller chunks are faster but lose sentence-level context across boundaries.
200 token overlap prevents answers from being cut off mid-sentence.

## Embedding Model: all-MiniLM-L6-v2
Chosen for CPU speed on free-tier hosting (HuggingFace Spaces).
Alternatives like all-mpnet-base-v2 are ~15% more accurate but 3x slower.
For a demo RAG app, speed-accuracy tradeoff favors MiniLM.

## Vector Store: ChromaDB (in-memory)
No persistent storage needed for single-session demo use.
Pinecone or Weaviate would be the production choice for multi-user, persistent indexes.
ChromaDB requires zero infrastructure setup, making local dev and HF Spaces deployment simple.

## Retriever: top-4 chunks (k=4)
k=3 occasionally misses edge-case context.
k=6 adds irrelevant chunks that confuse the LLM and degrade answer quality.
k=4 is the tested sweet spot for single-topic PDFs up to ~100 pages.

## LLM: Gemini 2.5 Flash
Chosen over GPT-4o for two reasons:
1. Free tier via Google AI Studio — no billing required for demo use.
2. Speed: Flash variant has lower latency than Pro, acceptable for Q&A tasks.
Quality is sufficient for document Q&A where the context is explicitly provided via RAG.

## Memory: ConversationBufferMemory
Full conversation history passed each turn.
Acceptable for short sessions. For long conversations (20+ turns),
ConversationSummaryMemory would be the better choice to avoid token limit issues.
