"""
Ablation Study & Evaluation Benchmark for PDF RAG Chatbot.

Evaluates 4 retrieval architectures across 40 standardized domain queries:
1. BM25 Only (Sparse Lexical Search)
2. Vector Only (ChromaDB Dense Embedding Search)
3. Hybrid Search (EnsembleRetriever: Dense 0.6 + BM25 0.4)
4. Hybrid + Reranking (EnsembleRetriever + CrossEncoder ms-marco-MiniLM-L-6-v2)

Metrics Evaluated:
- RAGAS Faithfulness
- Retrieval Context Hit Rate (Ground Truth Retrieval @ k)
- End-to-End Latency per query
"""

import os
import io
import json
import time
from typing import List, Dict, Any
from fpdf import FPDF
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

# Fix Ragas hardcoded import bug with Langchain 0.3+
import sys
import types
mock_module = types.ModuleType("langchain_community.chat_models.vertexai")
mock_module.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = mock_module

from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.pdf_processor import build_retrievers
from utils.chat_engine import get_conversation_chain

load_dotenv()


class MockUploadedFile:
    """Mock Streamlit UploadedFile from in-memory bytes."""
    def __init__(self, name: str, stream: io.BytesIO):
        self.name = name
        self.stream = stream

    def read(self, *args, **kwargs):
        return self.stream.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.stream.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.stream.tell(*args, **kwargs)


def create_domain_pdf(filename: str, title: str, paragraphs: List[str]) -> MockUploadedFile:
    """Generate in-memory PDF from text paragraphs."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt=title, ln=True, align='L')
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    for p in paragraphs:
        # Encode as latin-1 safe string
        safe_p = p.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, txt=safe_p)
        pdf.ln(4)

    out = pdf.output(dest='S')
    pdf_bytes = bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode('latin1')
    return MockUploadedFile(filename, io.BytesIO(pdf_bytes))


# ==============================================================================
# 4 DOMAIN CORPUS (Finance, Clinical Health, Structural Engineering, Cloud SLA)
# ==============================================================================

DOC_FINANCE = {
    "filename": "acme_q3_financial_report.pdf",
    "title": "Acme Corp - Q3 FY2024 Financial Performance & SEC 10-Q Summary",
    "paragraphs": [
        "Acme Corp reported consolidated revenue of $45.2 million for the third quarter of fiscal 2024, representing a 12.4% year-over-year increase compared to $40.2 million in Q2. Operating income reached $11.8 million, delivering an operating margin of 26.1%, an expansion of 210 basis points over the prior quarter.",
        "Operating expenses decreased by $2.3 million to $19.4 million, driven primarily by automated cloud infrastructure consolidation and vendor renegotiations. Research and Development (R&D) investments totaled $8.4 million, representing 18.6% of total revenue, focused on generative AI workflows.",
        "Enterprise Annual Recurring Revenue (ARR) expanded to $58.0 million with a Net Revenue Retention (NRR) rate of 114%. The company generated $16.5 million in net cash from operating activities, with free cash flow standing at $13.4 million after capital expenditures of $3.1 million for edge server hardware.",
        "The Board of Directors authorized an expanded share repurchase program of up to $25.0 million commencing on November 15, 2024. Total cash, cash equivalents, and marketable securities stood at $94.5 million as of September 30, 2024. Full-year FY2024 revenue guidance was revised upward to between $180 million and $185 million."
    ]
}

DOC_HEALTH = {
    "filename": "xr2_vaccine_clinical_protocol.pdf",
    "title": "XR-2 mRNA Vaccine - Phase III Clinical Protocol & Safety Summary",
    "paragraphs": [
        "The XR-2 prophylactic mRNA vaccine is indicated for active immunization against novel viral respiratory distress. The standard clinical regimen consists of two intramuscular doses of 0.5 mL each, administered precisely 21 days apart into the deltoid muscle.",
        "Phase III double-blind randomized clinical trial results demonstrated an overall vaccine efficacy of 94.6% against symptomatic infection among 32,450 evaluable participants. Neutralizing antibody titers peaked at day 42 post-initial dose and remained elevated above protective thresholds at 6-month follow-up.",
        "Cold-chain storage specifications dictate ultra-low frozen storage between -80 degrees Celsius and -60 degrees Celsius for up to 9 months. Vials may be transferred to standard refrigeration (2 to 8 degrees Celsius) for a maximum duration of 30 days prior to reconstitution. Once punctured, unpreserved vials must be administered within 6 hours at room temperature (15 to 25 degrees Celsius).",
        "The safety profile was characterized predominantly by mild-to-moderate transient adverse reactions: injection site pain (84.1%), general fatigue (59.2%), headache (52.3%), and low-grade pyrexia / mild fever (14.2%), typically resolving within 48 to 72 hours. Anaphylaxis occurred at a rate of 2.5 per 1,000,000 administered doses. Administration is strictly contraindicated in patients with severe hypersensitivity to polyethylene glycol (PEG)."
    ]
}

DOC_ENGINEERING = {
    "filename": "alpha_bridge_structural_specs.pdf",
    "title": "Alpha Suspension Bridge - Structural Engineering Specifications & Health Assessment",
    "paragraphs": [
        "The Alpha Suspension Bridge spans a main channel length of 1,280 meters. The superstructure is suspended from two 920 mm diameter high-tensile carbon-fiber reinforced steel main cables, each manufactured with ASTM A709 Grade 50W weathering steel to resist marine atmospheric corrosion.",
        "The maximum continuous design live-load capacity is 4,500 metric tons, with an instantaneous ultimate limit state threshold of 6,200 metric tons. The deck cross-section features streamlined aerodynamic fairings and center median open-grating designed to suppress flutter under sustained hurricane-force winds up to 210 km/h and gust velocities up to 260 km/h.",
        "Seismic protection is provided by triple-friction pendulum base-isolation bearings installed on the main tower footings, engineered to absorb ground accelerations up to Magnitude 8.2 on the Richter scale. Tower piers extend 45 meters below mean sea level and are socketed directly into granitic bedrock.",
        "Structural health monitoring consists of 420 fiber-optic Bragg grating strain sensors continuously polling at 100 Hz, coupled with two 150-ton tuned mass dampers inside the north and south towers. Mandatory non-destructive ultrasonic thickness testing and magnetic particle weld inspections are executed at strict 6-month intervals. Protective elastomeric cable coating requires reapplication every 12 years."
    ]
}

DOC_CLOUD = {
    "filename": "acme_cloud_storage_sla.pdf",
    "title": "Acme Cloud Distributed Object Storage - Architecture & SLA Agreement",
    "paragraphs": [
        "Acme Cloud Object Store provides distributed object storage with designed data durability of 99.999999999% (11 nines) per annum, achieved through 8+4 Reed-Solomon erasure coding across three distinct availability zones. Monthly service availability is guaranteed at 99.99% under the enterprise Service Level Agreement (SLA).",
        "Performance benchmarks commit to a median (P50) read latency of 12 milliseconds and a 99th percentile (P99) read latency under 45 milliseconds for payload sizes up to 10 MB. Individual object sizes range from 1 byte up to a maximum of 5 Terabytes via the multipart chunked upload API with 5 MB minimum part size.",
        "Security architecture mandates default server-side encryption using AES-256-GCM, supporting customer-managed encryption keys (CMEK) integrated with external Key Management Systems. Network ingress strictly requires TLS 1.3 with forward secrecy; legacy SSLv3 and TLS 1.0/1.1 connections are automatically dropped at the border gateway.",
        "Disaster recovery architecture maintains synchronous cross-zone replication with a Recovery Point Objective (RPO) of zero seconds and an automated DNS failover Recovery Time Objective (RTO) of under 60 seconds. Automated lifecycle rules transition objects unaccessed for 90 days to cold glacier archive, cutting storage costs by 68%. Bucket throughput is rate-limited to 5,000 read QPS and 2,000 write QPS per tenant."
    ]
}

# ==============================================================================
# 40 STANDARDIZED EVALUATION QUERIES (10 per Domain, Documented Provenance)
# ==============================================================================

BENCHMARK_QUERIES = [
    # Domain 1: Finance (10 queries)
    {
        "id": "FIN-01",
        "domain": "Finance",
        "type": "Factual Extraction",
        "q": "What was Acme Corp's reported revenue in Q3 FY2024?",
        "ground_truth": "$45.2 million"
    },
    {
        "id": "FIN-02",
        "domain": "Finance",
        "type": "Factual Extraction",
        "q": "What was the year-over-year revenue percentage increase from Q2 to Q3?",
        "ground_truth": "12.4% increase compared to $40.2 million in Q2"
    },
    {
        "id": "FIN-03",
        "domain": "Finance",
        "type": "Numeric Metric",
        "q": "What was the operating income and operating margin for Acme Corp in Q3?",
        "ground_truth": "Operating income was $11.8 million with an operating margin of 26.1%"
    },
    {
        "id": "FIN-04",
        "domain": "Finance",
        "type": "Causal / Explanation",
        "q": "Why did operating expenses decrease by $2.3 million?",
        "ground_truth": "Driven by automated cloud infrastructure consolidation and vendor renegotiations"
    },
    {
        "id": "FIN-05",
        "domain": "Finance",
        "type": "Numeric Metric",
        "q": "How much capital was spent on Research and Development (R&D) in Q3?",
        "ground_truth": "$8.4 million, representing 18.6% of total revenue"
    },
    {
        "id": "FIN-06",
        "domain": "Finance",
        "type": "Factual Extraction",
        "q": "What is Acme Corp's Enterprise Annual Recurring Revenue (ARR) and Net Revenue Retention?",
        "ground_truth": "Enterprise ARR reached $58.0 million with a 114% Net Revenue Retention rate"
    },
    {
        "id": "FIN-07",
        "domain": "Finance",
        "type": "Factual Extraction",
        "q": "When does the new $25 million share repurchase program start?",
        "ground_truth": "Commencing on November 15, 2024"
    },
    {
        "id": "FIN-08",
        "domain": "Finance",
        "type": "Numeric Metric",
        "q": "How much free cash flow was generated after server hardware capital expenditures?",
        "ground_truth": "$13.4 million in free cash flow after $3.1 million in capital expenditures"
    },
    {
        "id": "FIN-09",
        "domain": "Finance",
        "type": "Factual Extraction",
        "q": "What was total cash and cash equivalents at the end of September 2024?",
        "ground_truth": "$94.5 million as of September 30, 2024"
    },
    {
        "id": "FIN-10",
        "domain": "Finance",
        "type": "Forward Guidance",
        "q": "What is the revised full-year FY2024 revenue guidance range?",
        "ground_truth": "Between $180 million and $185 million"
    },

    # Domain 2: Health / Clinical (10 queries)
    {
        "id": "HLT-01",
        "domain": "Health",
        "type": "Factual Extraction",
        "q": "What is the required dosage and administration interval for the XR-2 vaccine?",
        "ground_truth": "Two intramuscular doses of 0.5 mL each, administered 21 days apart"
    },
    {
        "id": "HLT-02",
        "domain": "Health",
        "type": "Numeric Metric",
        "q": "What was the reported efficacy rate of the XR-2 vaccine in Phase III trials?",
        "ground_truth": "94.6% efficacy against symptomatic infection across 32,450 trial participants"
    },
    {
        "id": "HLT-03",
        "domain": "Health",
        "type": "Clinical Metric",
        "q": "At what day post-initial dose did neutralizing antibody titers reach their peak?",
        "ground_truth": "Day 42 post-initial dose"
    },
    {
        "id": "HLT-04",
        "domain": "Health",
        "type": "Storage Specification",
        "q": "What are the ultra-cold temperature storage requirements for long-term storage of XR-2?",
        "ground_truth": "Between -80 degrees Celsius and -60 degrees Celsius for up to 9 months"
    },
    {
        "id": "HLT-05",
        "domain": "Health",
        "type": "Storage Specification",
        "q": "How long can the XR-2 vaccine be kept at standard refrigeration temperatures (2 to 8°C)?",
        "ground_truth": "Maximum duration of 30 days"
    },
    {
        "id": "HLT-06",
        "domain": "Health",
        "type": "Operational Handling",
        "q": "Once punctured, within how many hours must an XR-2 vial be administered at room temperature?",
        "ground_truth": "Within 6 hours at room temperature (15 to 25 degrees Celsius)"
    },
    {
        "id": "HLT-07",
        "domain": "Health",
        "type": "Adverse Reactions",
        "q": "What was the most commonly reported adverse reaction and its percentage incidence?",
        "ground_truth": "Injection site pain at 84.1%"
    },
    {
        "id": "HLT-08",
        "domain": "Health",
        "type": "Clinical Resolution",
        "q": "How long do typical mild-to-moderate vaccine side effects take to resolve?",
        "ground_truth": "Resolving within 48 to 72 hours"
    },
    {
        "id": "HLT-09",
        "domain": "Health",
        "type": "Adverse Reactions",
        "q": "What was the incidence rate of anaphylaxis observed during the trial?",
        "ground_truth": "2.5 per 1,000,000 administered doses"
    },
    {
        "id": "HLT-10",
        "domain": "Health",
        "type": "Contraindication",
        "q": "Which patient group is strictly contraindicated from receiving the XR-2 vaccine?",
        "ground_truth": "Patients with severe hypersensitivity to polyethylene glycol (PEG)"
    },

    # Domain 3: Structural Engineering (10 queries)
    {
        "id": "ENG-01",
        "domain": "Engineering",
        "type": "Factual Extraction",
        "q": "What is the main channel span length of the Alpha Suspension Bridge?",
        "ground_truth": "1,280 meters"
    },
    {
        "id": "ENG-02",
        "domain": "Engineering",
        "type": "Material Spec",
        "q": "What steel grade and diameter are used for the main suspension cables?",
        "ground_truth": "920 mm diameter cables made with ASTM A709 Grade 50W weathering steel"
    },
    {
        "id": "ENG-03",
        "domain": "Engineering",
        "type": "Structural Limits",
        "q": "What is the maximum continuous live-load capacity of the Alpha Bridge?",
        "ground_truth": "4,500 metric tons, with an ultimate limit state of 6,200 metric tons"
    },
    {
        "id": "ENG-04",
        "domain": "Engineering",
        "type": "Aerodynamics",
        "q": "What maximum sustained wind speed and gust velocity is the aerodynamic deck rated to withstand?",
        "ground_truth": "Sustained winds up to 210 km/h and gust velocities up to 260 km/h"
    },
    {
        "id": "ENG-05",
        "domain": "Engineering",
        "type": "Seismic Design",
        "q": "What earthquake magnitude is the base-isolation bearing system engineered to absorb?",
        "ground_truth": "Up to Magnitude 8.2 on the Richter scale"
    },
    {
        "id": "ENG-06",
        "domain": "Engineering",
        "type": "Geotechnical Spec",
        "q": "How deep below mean sea level are the tower pier footings anchored?",
        "ground_truth": "45 meters below mean sea level socketed into granitic bedrock"
    },
    {
        "id": "ENG-07",
        "domain": "Engineering",
        "type": "Sensor Instrumentation",
        "q": "How many fiber-optic strain sensors monitor the Alpha Bridge and at what frequency?",
        "ground_truth": "420 fiber-optic Bragg grating sensors polling continuously at 100 Hz"
    },
    {
        "id": "ENG-08",
        "domain": "Engineering",
        "type": "Structural Dampening",
        "q": "What damping mechanisms mitigate dynamic wind sway in the north and south towers?",
        "ground_truth": "Two 150-ton tuned mass dampers installed inside the towers"
    },
    {
        "id": "ENG-09",
        "domain": "Engineering",
        "type": "Maintenance Schedule",
        "q": "How often are mandatory ultrasonic thickness tests and weld inspections required?",
        "ground_truth": "Every 6 months"
    },
    {
        "id": "ENG-10",
        "domain": "Engineering",
        "type": "Maintenance Schedule",
        "q": "What is the maintenance interval for recoating the protective elastomeric cable coating?",
        "ground_truth": "Every 12 years"
    },

    # Domain 4: Cloud Infrastructure & Security (10 queries)
    {
        "id": "CLD-01",
        "domain": "Cloud",
        "type": "Durability & Availability",
        "q": "What is the annual data durability design and monthly SLA uptime for Acme Cloud storage?",
        "ground_truth": "99.999999999% (11 nines) durability and 99.99% monthly SLA availability"
    },
    {
        "id": "CLD-02",
        "domain": "Cloud",
        "type": "Storage Architecture",
        "q": "What erasure coding scheme is used across availability zones?",
        "ground_truth": "8+4 Reed-Solomon erasure coding across three distinct availability zones"
    },
    {
        "id": "CLD-03",
        "domain": "Cloud",
        "type": "Latency SLA",
        "q": "What are the guaranteed P50 and P99 read latencies for payloads under 10 MB?",
        "ground_truth": "P50 read latency of 12 ms and P99 read latency under 45 ms"
    },
    {
        "id": "CLD-04",
        "domain": "Cloud",
        "type": "Object Specifications",
        "q": "What is the maximum single object size supported and the minimum multipart chunk size?",
        "ground_truth": "Maximum object size of 5 Terabytes with 5 MB minimum multipart chunk size"
    },
    {
        "id": "CLD-05",
        "domain": "Cloud",
        "type": "Encryption Standard",
        "q": "What cryptographic algorithm is enforced for default server-side encryption at rest?",
        "ground_truth": "AES-256-GCM with customer-managed encryption keys (CMEK)"
    },
    {
        "id": "CLD-06",
        "domain": "Cloud",
        "type": "Network Security",
        "q": "What transport security protocol is mandatory for network ingress, and which are blocked?",
        "ground_truth": "TLS 1.3 is mandatory; legacy SSLv3 and TLS 1.0/1.1 are dropped at border gateway"
    },
    {
        "id": "CLD-07",
        "domain": "Cloud",
        "type": "Disaster Recovery",
        "q": "What are the Recovery Point Objective (RPO) and Recovery Time Objective (RTO) targets?",
        "ground_truth": "RPO of zero seconds and automated DNS failover RTO under 60 seconds"
    },
    {
        "id": "CLD-08",
        "domain": "Cloud",
        "type": "Lifecycle Management",
        "q": "After how many days are unaccessed objects moved to cold archive, and how much cost is saved?",
        "ground_truth": "Transitioned after 90 days, cutting storage costs by 68%"
    },
    {
        "id": "CLD-09",
        "domain": "Cloud",
        "type": "Rate Limiting",
        "q": "What are the per-tenant read and write throughput rate limits in requests per second?",
        "ground_truth": "5,000 read QPS and 2,000 write QPS per tenant"
    },
    {
        "id": "CLD-10",
        "domain": "Cloud",
        "type": "Resilience / Failover",
        "q": "How does Acme Cloud execute disaster recovery across regions?",
        "ground_truth": "Synchronous cross-zone replication with automated DNS failover in under 60 seconds"
    }
]


def check_hit_rate(retrieved_docs: List[Any], ground_truth: str) -> float:
    """Check if any retrieved doc contains key semantic tokens of the ground truth."""
    gt_tokens = [tok.lower().strip("$,%.()") for tok in ground_truth.split() if len(tok) > 3]
    if not gt_tokens:
        return 1.0
    for doc in retrieved_docs:
        content = doc.page_content.lower()
        # If at least 50% of the significant tokens are in this chunk, it's a hit
        matches = sum(1 for tok in gt_tokens if tok in content)
        if matches / len(gt_tokens) >= 0.5:
            return 1.0
    return 0.0


def run_ablation_study():
    """Execute ablation benchmark across all 4 architectures on 40 queries."""
    print("=" * 70)
    print("STARTING ABLATION BENCHMARK (4 Architectures x 40 Queries)")
    print("=" * 70)

    # 1. Create Domain PDFs
    print("\n[Phase 1/5] Synthesizing 4 Multi-Domain Evaluation PDFs...")
    pdf_docs = [
        create_domain_pdf(DOC_FINANCE["filename"], DOC_FINANCE["title"], DOC_FINANCE["paragraphs"]),
        create_domain_pdf(DOC_HEALTH["filename"], DOC_HEALTH["title"], DOC_HEALTH["paragraphs"]),
        create_domain_pdf(DOC_ENGINEERING["filename"], DOC_ENGINEERING["title"], DOC_ENGINEERING["paragraphs"]),
        create_domain_pdf(DOC_CLOUD["filename"], DOC_CLOUD["title"], DOC_CLOUD["paragraphs"])
    ]

    # 2. Build Retrievers
    print("[Phase 2/5] Initializing Retrievers (BM25, Chroma Dense, Ensemble)...")
    retrievers = build_retrievers(pdf_docs)
    dense_ret = retrievers["dense"]
    sparse_ret = retrievers["sparse"]
    ensemble_ret = retrievers["ensemble"]

    # 3. LLM Setup (Qwen 27B / Gemini 2.5 Flash)
    print("[Phase 3/5] Initializing LLM Chains...")
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if groq_key:
        llm = ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0.0, api_key=groq_key)
    elif gemini_key:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=gemini_key)
    else:
        raise ValueError("Missing API key in environment.")

    # 4 Architecture Configurations:
    # 1. BM25 Only: sparse retriever, no reranker
    # 2. Vector Only: dense retriever, no reranker
    # 3. Hybrid: ensemble retriever, no reranker
    # 4. Hybrid + Rerank: ensemble retriever, with CrossEncoder reranker
    configs = {
        "bm25_only": {
            "name": "BM25 Sparse Only",
            "chain": get_conversation_chain(sparse_ret, llm=llm, use_reranker=False),
            "retriever": sparse_ret
        },
        "vector_only": {
            "name": "Vector Dense Only (Chroma)",
            "chain": get_conversation_chain(dense_ret, llm=llm, use_reranker=False),
            "retriever": dense_ret
        },
        "hybrid": {
            "name": "Hybrid (Dense + BM25)",
            "chain": get_conversation_chain(ensemble_ret, llm=llm, use_reranker=False),
            "retriever": ensemble_ret
        },
        "hybrid_rerank": {
            "name": "Hybrid + CrossEncoder Reranking",
            "chain": get_conversation_chain(ensemble_ret, llm=llm, use_reranker=True),
            "retriever": ensemble_ret
        }
    }

    # Setup RAGAS Evaluator
    print("[Phase 4/5] Configuring RAGAS Evaluator...")
    groq_client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    eval_llm = llm_factory("qwen/qwen3.8-27b", client=groq_client)
    faithfulness.llm = eval_llm

    results_summary = {}

    # Run queries per configuration
    for cfg_key, cfg in configs.items():
        print(f"\n--- Testing Pipeline: {cfg['name']} ---")
        chain = cfg["chain"]
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        hit_rates = []
        latencies = []

        for idx, item in enumerate(BENCHMARK_QUERIES, 1):
            q = item["q"]
            gt = item["ground_truth"]
            t0 = time.time()
            try:
                res = chain.invoke({"input": q, "chat_history": []})
                latency = time.time() - t0
                ans = res.get("answer", "")
                docs = res.get("context", [])
                ctx = [d.page_content for d in docs]
                hit = check_hit_rate(docs, gt)
            except Exception as e:
                print(f"  [!] Error on Q{idx}: {e}")
                ans = "Error generating answer"
                ctx = []
                hit = 0.0
                latency = time.time() - t0

            questions.append(q)
            answers.append(ans)
            contexts_list.append(ctx)
            ground_truths.append(gt)
            hit_rates.append(hit)
            latencies.append(latency)

            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(BENCHMARK_QUERIES)} queries...")

        # Compute RAGAS Faithfulness on this pipeline (no silent fallbacks; errors must fail loudly)
        print(f"  Computing RAGAS Faithfulness for {cfg['name']}...")
        ds = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        })

        ragas_res = evaluate(
            dataset=ds,
            metrics=[faithfulness],
            llm=eval_llm,
            raise_exceptions=True,
            run_config=RunConfig(max_workers=2, max_retries=10, max_wait=30)
        )
        df_ragas = ragas_res.to_pandas()
        mean_faithfulness = float(df_ragas["faithfulness"].fillna(0.0).mean())

        mean_hit_rate = float(sum(hit_rates) / len(hit_rates))
        mean_latency = float(sum(latencies) / len(latencies))

        results_summary[cfg_key] = {
            "name": cfg["name"],
            "faithfulness": round(mean_faithfulness, 4),
            "retrieval_hit_rate": round(mean_hit_rate, 4),
            "avg_latency_seconds": round(mean_latency, 2),
            "sample_size": len(BENCHMARK_QUERIES)
        }
        print(f"  => Faithfulness: {mean_faithfulness:.4f} | Hit Rate: {mean_hit_rate*100:.1f}% | Avg Latency: {mean_latency:.2f}s")

    # 5. Output Summary
    print("\n" + "=" * 70)
    print("FINAL ABLATION STUDY RESULTS")
    print("=" * 70)
    print(json.dumps(results_summary, indent=4))

    vector_f = results_summary["vector_only"]["faithfulness"]
    hybrid_rerank_f = results_summary["hybrid_rerank"]["faithfulness"]
    f_diff = hybrid_rerank_f - vector_f
    f_pct = (f_diff / vector_f) * 100 if vector_f > 0 else 0.0

    vector_hit = results_summary["vector_only"]["retrieval_hit_rate"]
    hybrid_rerank_hit = results_summary["hybrid_rerank"]["retrieval_hit_rate"]
    hit_diff = (hybrid_rerank_hit - vector_hit) * 100

    output_payload = {
        "metadata": {
            "benchmark": "Synthetic Multi-Domain Benchmark (4 Fictional Documents)",
            "benchmark_nature": "synthetic",
            "sample_size": len(BENCHMARK_QUERIES),
            "domains": ["Finance (Fictional SEC 10-Q)", "Clinical Health (Fictional mRNA Trial)", "Structural Engineering (Fictional Bridge)", "Cloud Infrastructure SLA (Fictional Spec)"],
            "previous_baseline": {
                "faithfulness": 0.9667,
                "samples": 15,
                "note": "Initial synthetic exploratory set"
            }
        },
        "ablation_results": results_summary,
        "metrics_summary": {
            "vector_faithfulness": vector_f,
            "hybrid_rerank_faithfulness": hybrid_rerank_f,
            "faithfulness_difference": round(f_diff, 4),
            "hit_rate_difference_percentage_points": round(hit_diff, 1)
        }
    }

    with open("eval_ablation_results.json", "w") as f:
        json.dump(output_payload, f, indent=4)
    print("\nResults successfully saved to eval_ablation_results.json")

    # Also update eval_results.json with the new primary hybrid_rerank metrics while retaining history
    with open("eval_results.json", "w") as f:
        json.dump({
            "current": {
                "faithfulness": hybrid_rerank_f,
                "retrieval_hit_rate": hybrid_rerank_hit,
                "evaluated_samples": len(BENCHMARK_QUERIES),
                "evaluation_model": "qwen/qwen3.8-27b + ms-marco-MiniLM-L-6-v2",
                "domains": ["Finance", "Clinical Health", "Structural Engineering", "Cloud Infrastructure"]
            },
            "history": {
                "initial_baseline": {
                    "faithfulness": 0.9667,
                    "evaluated_samples": 15,
                    "note": "Initial 15-sample prototype evaluation"
                }
            }
        }, f, indent=4)
    print("Updated eval_results.json with current results and historical baseline.")


if __name__ == "__main__":
    run_ablation_study()
