"""Unit tests for dense, sparse, ensemble retrieval and reranking."""

import pytest
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.pdf_processor import build_retrievers
from utils.chat_engine import get_conversation_chain, load_cross_encoder


class TestRetrievalPipeline:
    """Test suite for RAG retrieval components."""

    def test_bm25_sparse_retrieval(self, sample_single_pdf):
        """Verify BM25 sparse keyword retriever identifies exact lexical matches."""
        retrievers = build_retrievers([sample_single_pdf])
        sparse_ret = retrievers["sparse"]
        results = sparse_ret.invoke("operating margin")
        assert len(results) > 0
        combined = " ".join([d.page_content for d in results])
        assert "26.1%" in combined

    def test_dense_vector_retrieval(self, sample_single_pdf):
        """Verify ChromaDB dense vector store retrieves relevant semantic context."""
        retrievers = build_retrievers([sample_single_pdf])
        dense_ret = retrievers["dense"]
        results = dense_ret.invoke("How much money did the company make?")
        assert len(results) > 0
        combined = " ".join([d.page_content for d in results])
        assert "45.2 million" in combined or "revenue" in combined.lower()

    def test_ensemble_hybrid_retrieval(self, sample_multi_pdfs):
        """Verify EnsembleRetriever merges results from multiple documents."""
        retrievers = build_retrievers(sample_multi_pdfs)
        ensemble_ret = retrievers["ensemble"]
        assert isinstance(ensemble_ret, EnsembleRetriever)

        # Query across multi-domain docs
        res_health = ensemble_ret.invoke("vaccine doses interval")
        assert len(res_health) > 0
        assert any("21 days" in d.page_content for d in res_health)

        res_bridge = ensemble_ret.invoke("bridge span length steel")
        assert len(res_bridge) > 0
        assert any("1,280 meters" in d.page_content or "weathering steel" in d.page_content for d in res_bridge)

    def test_cross_encoder_reranker_filtering(self, sample_multi_pdfs):
        """Verify CrossEncoder model loads and reranker initializes properly."""
        encoder = load_cross_encoder()
        assert encoder is not None
        # Score query against two sentences (one relevant, one irrelevant)
        scores = encoder.score([
            ("vaccine doses and efficacy", "The XR-2 vaccine requires two doses with 94.6% efficacy."),
            ("vaccine doses and efficacy", "The bridge cables are made with ASTM weathering steel.")
        ])
        assert len(scores) == 2
        # Health sentence must score higher than bridge sentence
        assert scores[0] > scores[1]

    def test_conversation_chain_builder_with_and_without_reranker(self, sample_single_pdf):
        """Verify get_conversation_chain supports both reranked and non-reranked modes."""
        retrievers = build_retrievers([sample_single_pdf])
        retriever = retrievers["ensemble"]

        # 1. With reranker
        chain_with_rerank = get_conversation_chain(retriever, use_reranker=True)
        assert chain_with_rerank is not None

        # 2. Without reranker (ablation baseline)
        chain_without_rerank = get_conversation_chain(retriever, use_reranker=False)
        assert chain_without_rerank is not None
