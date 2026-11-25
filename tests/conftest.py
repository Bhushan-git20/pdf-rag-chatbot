"""Pytest test fixtures for PDF RAG Chatbot test suite."""

import sys
import os
from pathlib import Path

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import io
import pytest
from fpdf import FPDF


class MockFile:
    """Mock uploaded file object matching Streamlit UploadedFile interface."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self._stream = io.BytesIO(data)

    def read(self, *args, **kwargs):
        return self._stream.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._stream.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self._stream.tell(*args, **kwargs)


@pytest.fixture
def sample_single_pdf():
    """Generates a standard single valid text PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Acme Corporation Q3 Financial Report.", ln=True)
    pdf.multi_cell(0, 8, "Total revenue reached $45.2 million, an increase of 12.4% over Q2. Operating margin was 26.1%.")
    out = pdf.output(dest='S')
    pdf_bytes = bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode('latin1')
    return MockFile("report_single.pdf", pdf_bytes)


@pytest.fixture
def sample_multi_pdfs():
    """Generates a list of multiple valid PDFs across distinct domains."""
    # PDF 1: Medical / Clinical
    pdf1 = FPDF()
    pdf1.add_page()
    pdf1.set_font("Arial", size=12)
    pdf1.cell(0, 10, "XR-2 Clinical Protocol", ln=True)
    pdf1.multi_cell(0, 8, "The XR-2 vaccine requires two doses separated by 21 days with 94.6% efficacy.")
    out1 = pdf1.output(dest='S')
    b1 = bytes(out1) if isinstance(out1, (bytes, bytearray)) else out1.encode('latin1')

    # PDF 2: Engineering
    pdf2 = FPDF()
    pdf2.add_page()
    pdf2.set_font("Arial", size=12)
    pdf2.cell(0, 10, "Alpha Bridge Specs", ln=True)
    pdf2.multi_cell(0, 8, "The Alpha Bridge spans 1,280 meters and uses ASTM A709 weathering steel cables.")
    out2 = pdf2.output(dest='S')
    b2 = bytes(out2) if isinstance(out2, (bytes, bytearray)) else out2.encode('latin1')

    return [MockFile("health.pdf", b1), MockFile("engineering.pdf", b2)]


@pytest.fixture
def sample_table_pdf():
    """Generates a PDF containing structured columnar table text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 10, "Infrastructure Latency Benchmarks (Table)", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(40, 8, "Region", border=1)
    pdf.cell(40, 8, "P50 Latency", border=1)
    pdf.cell(40, 8, "P99 Latency", border=1)
    pdf.ln()
    pdf.cell(40, 8, "us-east-1", border=1)
    pdf.cell(40, 8, "12 ms", border=1)
    pdf.cell(40, 8, "45 ms", border=1)
    pdf.ln()
    pdf.cell(40, 8, "eu-west-1", border=1)
    pdf.cell(40, 8, "15 ms", border=1)
    pdf.cell(40, 8, "52 ms", border=1)
    out = pdf.output(dest='S')
    pdf_bytes = bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode('latin1')
    return MockFile("benchmark_table.pdf", pdf_bytes)


@pytest.fixture
def sample_empty_pdf():
    """Generates an empty page PDF (simulating scanned/image document with no text layer)."""
    pdf = FPDF()
    pdf.add_page()
    out = pdf.output(dest='S')
    pdf_bytes = bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode('latin1')
    return MockFile("empty_scanned.pdf", pdf_bytes)


@pytest.fixture
def sample_corrupted_pdf():
    """Generates an invalid/corrupted binary payload pretending to be a PDF."""
    corrupted_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nThis is completely corrupt random garbage data that cannot be parsed."
    return MockFile("corrupted_payload.pdf", corrupted_bytes)
