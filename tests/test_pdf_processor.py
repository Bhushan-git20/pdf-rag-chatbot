"""Unit tests for PDF extraction, splitting, and corrupted file handling."""

import pytest
from utils.pdf_processor import extract_text_from_pdfs, split_documents, process_pdfs


class TestPdfProcessor:
    """Test suite for PDF processing pipeline."""

    def test_extract_text_single_pdf(self, sample_single_pdf):
        """Verify successful text extraction from a single valid PDF."""
        extracted = extract_text_from_pdfs([sample_single_pdf])
        assert len(extracted) >= 1
        assert "Acme Corporation Q3 Financial Report" in extracted[0]["text"]
        assert extracted[0]["source"] == "report_single.pdf (Page 1)"

    def test_extract_text_multi_pdfs(self, sample_multi_pdfs):
        """Verify text extraction across multiple uploaded PDFs."""
        extracted = extract_text_from_pdfs(sample_multi_pdfs)
        assert len(extracted) == 2
        sources = [item["source"] for item in extracted]
        assert "health.pdf (Page 1)" in sources
        assert "engineering.pdf (Page 1)" in sources

    def test_extract_text_table_pdf(self, sample_table_pdf):
        """Verify text extraction preserves columnar table content."""
        extracted = extract_text_from_pdfs([sample_table_pdf])
        assert len(extracted) >= 1
        text = extracted[0]["text"]
        assert "us-east-1" in text
        assert "12 ms" in text
        assert "eu-west-1" in text

    def test_corrupted_pdf_graceful_handling(self, sample_corrupted_pdf):
        """Verify corrupted binary file is caught gracefully and does not crash."""
        extracted = extract_text_from_pdfs([sample_corrupted_pdf])
        # Corrupted PDF should yield no extracted text blocks without raising uncaught exceptions
        assert extracted == []

    def test_corrupted_and_valid_mixed_pdfs(self, sample_single_pdf, sample_corrupted_pdf):
        """Verify system extracts valid PDFs even if one file is corrupted."""
        extracted = extract_text_from_pdfs([sample_corrupted_pdf, sample_single_pdf])
        assert len(extracted) == 1
        assert "Acme Corporation" in extracted[0]["text"]

    def test_empty_or_scanned_pdf_raises_value_error(self, sample_empty_pdf):
        """Verify that attempting to process an empty/scanned PDF raises ValueError."""
        with pytest.raises(ValueError, match="No text extracted from PDFs"):
            process_pdfs([sample_empty_pdf])

    def test_split_documents_chunks_and_metadata(self, sample_single_pdf):
        """Verify recursive text splitting, min-chunk threshold, and source attribution."""
        raw_texts = extract_text_from_pdfs([sample_single_pdf])
        docs = split_documents(raw_texts)
        assert len(docs) >= 1
        for doc in docs:
            assert len(doc.page_content.strip()) > 10
            assert "source" in doc.metadata
            assert "report_single.pdf" in doc.metadata["source"]
