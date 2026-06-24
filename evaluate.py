import os
import io
import json
import time
from fpdf import FPDF
from dotenv import load_dotenv

import pandas as pd
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# MOCK missing module to fix Ragas hardcoded import bug with Langchain 0.3+
import sys
import types
mock_module = types.ModuleType("langchain_community.chat_models.vertexai")
mock_module.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = mock_module

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics.collections import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from utils.pdf_processor import process_pdfs
from utils.chat_engine import get_conversation_chain

load_dotenv()

def create_domain_pdf(filename, text_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text_content)
    
    out = pdf.output(dest='S')
    if isinstance(out, (bytes, bytearray)):
        pdf_bytes = bytes(out)
    else:
        pdf_bytes = out.encode('latin1')
    
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

    return MockUploadedFile(filename, io.BytesIO(pdf_bytes))

# Domain Texts
finance_text = "The Acme Corp Q3 revenue was $45 million, a 12% increase from Q2. Operating expenses were reduced by $2 million. The CEO announced a new stock buyback program starting in November."
health_text = "The new XR-2 vaccine requires two doses separated by 21 days. Side effects include mild fever and arm soreness. Storage must be maintained between 2 to 8 degrees Celsius."
engineering_text = "The structural integrity of the Alpha bridge relies on carbon-fiber reinforced steel cables. Maximum load capacity is 500 tons. Maintenance inspections are required every 6 months."

# 15 Questions
qa_pairs = [
    {"q": "What was Acme Corp's Q3 revenue?"},
    {"q": "What are the common side effects of the XR-2 vaccine?"},
    {"q": "What is the maximum load capacity of the Alpha bridge?"},
    {"q": "How much did Acme Corp's revenue increase from Q2?"},
    {"q": "What is the required storage temperature for the XR-2 vaccine?"}
]

def run_evaluation():
    print("1. Generating Domain PDFs...")
    f_pdf = create_domain_pdf("finance.pdf", finance_text)
    h_pdf = create_domain_pdf("health.pdf", health_text)
    e_pdf = create_domain_pdf("engineering.pdf", engineering_text)
    
    print("2. Processing PDFs & Initializing RAG Pipeline...")
    retriever = process_pdfs([f_pdf, h_pdf, e_pdf])
    chain = get_conversation_chain(retriever)
    
    questions = []
    answers = []
    contexts = []
    
    print("3. Querying RAG Pipeline (Fetching 15 answers)...")
    for item in qa_pairs:
        question = item["q"]
        print(f"  -> Q: {question}")
        res = chain.invoke({"input": question, "chat_history": []})
        
        answer = res.get("answer", "")
        # Ragas expects contexts as a list of strings
        source_texts = [doc.page_content for doc in res.get("context", [])]
        
        questions.append(question)
        answers.append(answer)
        contexts.append(source_texts)
        
        time.sleep(15) # Prevent extreme rate limiting during generation
        
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    print("4. Configuring RAGAS with Groq...")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=os.getenv("GEMINI_API_KEY"))
    
    eval_llm = LangchainLLMWrapper(llm)
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    print("5. Running RAGAS Evaluation (Faithfulness & Answer Relevance)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1, max_retries=20, max_wait=60)
    )
    
    print("\n--- EVALUATION COMPLETE ---")
    
    df = result.to_pandas()
    metrics_summary = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "evaluated_samples": len(df)
    }
    
    print(f"Faithfulness: {metrics_summary['faithfulness']:.2f} | Answer Relevance: {metrics_summary['answer_relevancy']:.2f}")
    print(f"Evaluated on {metrics_summary['evaluated_samples']} questions across 3 domain PDFs")
    
    with open("eval_results.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Saved summary to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
