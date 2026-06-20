import os
import io
import time
import sys
import traceback
from fpdf import FPDF
from PyPDF2 import PdfWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_processor import process_pdfs
from utils.chat_engine import get_conversation_chain
from dotenv import load_dotenv

load_dotenv()

results = []

def log(test_name, status, details=""):
    results.append({"test": test_name, "status": status, "details": details})
    print(f"[{status}] {test_name}: {details}")

class MockUploadedFile:
    def __init__(self, name, stream):
        self.name = name
        self.stream = stream

    def read(self, *args, **kwargs):
        return self.stream.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.stream.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.stream.tell(*args, **kwargs)

def create_pdf_with_text(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    
    out = pdf.output(dest='S')
    if isinstance(out, (bytes, bytearray)):
        pdf_bytes = bytes(out)
    else:
        pdf_bytes = out.encode('latin1')
    return io.BytesIO(pdf_bytes)

def create_empty_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    packet = io.BytesIO()
    writer.write(packet)
    packet.seek(0)
    return packet

def create_corrupt_pdf():
    return io.BytesIO(b"This is not a valid PDF file content. It should fail parsing.")

def run_tests():
    print("Starting Deep Stress Tests...\n")
    
    # 1. Test Valid PDF & Relevant Query
    try:
        print("--- Test 1: Valid PDF & Queries ---")
        pdf_stream = create_pdf_with_text("The secret password to the server is Banana123. The server IP is 192.168.1.100.")
        file1 = MockUploadedFile("secret.pdf", pdf_stream)
        
        retriever = process_pdfs([file1])
        log("Process Valid PDF", "PASS", "Retriever created successfully")
        
        chain = get_conversation_chain(retriever)
        log("Create Conversation Chain", "PASS", "Chain initialized successfully")
        
        print("-> Testing Relevant Query...")
        start_time = time.time()
        res = chain.invoke({"input": "What is the secret password?", "chat_history": []})
        duration = time.time() - start_time
        answer = res.get("answer", "")
        sources = res.get("context", [])
        
        if "Banana123" in answer and len(sources) > 0:
            log("Relevant Query", "PASS", f"Got correct answer in {duration:.2f}s")
        else:
            log("Relevant Query", "FAIL", f"Answer did not contain expected text. Answer: {answer}")
            
        print("-> Testing Irrelevant Query...")
        res_irr = chain.invoke({"input": "What is the capital of France? Only answer from the context provided.", "chat_history": []})
        ans_irr = res_irr.get("answer", "")
        src_irr = res_irr.get("context", [])
        
        if len(src_irr) == 0:
            log("Irrelevant Query (No sources)", "PASS", "Retriever found 0 sources as expected.")
        else:
            log("Irrelevant Query", "WARNING", f"Found {len(src_irr)} sources for irrelevant query. LLM answer: {ans_irr}")
            
    except Exception as e:
        log("Process Valid PDF", "FAIL", str(e))
        traceback.print_exc()

    print("\n--- Test 2: Empty PDF (Images only / No Extractable Text) ---")
    try:
        file2 = MockUploadedFile("empty.pdf", create_empty_pdf())
        retriever = process_pdfs([file2])
        log("Process Empty PDF", "FAIL", "Should have raised ValueError")
    except ValueError as e:
        log("Process Empty PDF", "PASS", f"Correctly caught: {str(e)}")
    except Exception as e:
        log("Process Empty PDF", "FAIL", f"Wrong exception type: {str(e)}")

    print("\n--- Test 3: Corrupt PDF / Non-PDF file ---")
    try:
        file3 = MockUploadedFile("corrupt.pdf", create_corrupt_pdf())
        retriever = process_pdfs([file3])
        log("Process Corrupt PDF", "FAIL", "Should have raised Exception")
    except Exception as e:
        log("Process Corrupt PDF", "PASS", f"Correctly caught exception during PDF read: {str(e)}")

    print("\n--- Test 4: Volume/Stress Testing (Large PDF & Multiple PDFs) ---")
    try:
        large_text = "Volume testing. Data block. " * 5000 # ~140k chars
        file_large = MockUploadedFile("large.pdf", create_pdf_with_text(large_text))
        file_small = MockUploadedFile("small.pdf", create_pdf_with_text("Just a small file with one sentence."))
        
        start_time = time.time()
        retriever = process_pdfs([file_large, file_small])
        duration = time.time() - start_time
        
        log("Process Large Volume PDFs", "PASS", f"Processed 2 files in {duration:.2f}s. Retriever initialized.")
        
        print("-> Testing Query on Large Volume...")
        chain_large = get_conversation_chain(retriever)
        start_time_q = time.time()
        res_large = chain_large.invoke({"input": "What does the small file say?", "chat_history": []})
        dur_q = time.time() - start_time_q
        
        log("Query on Large Volume", "PASS", f"Answered in {dur_q:.2f}s. Answer: {res_large.get('answer','')}")
        
    except Exception as e:
        log("Process Large Volume PDFs", "FAIL", str(e))
        traceback.print_exc()
        
    print("\n--- TEST SUMMARY ---")
    for r in results:
        print(f"[{r['status']}] {r['test']}: {r['details']}")

if __name__ == '__main__':
    run_tests()
