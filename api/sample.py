"""
Vercel Serverless Function: /api/sample
Returns sample document content for invoice and resume.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_INVOICE_FALLBACK = """ACME CLOUD SERVICES LLC
123 Innovation Way, Suite 400
Tech City, CA 94016
Email: billing@acmecloud.com

INVOICE

Invoice Number: INV-2024-00892
Invoice Date: October 15, 2024
Payment Due Date: November 14, 2024

BILLED TO:
Apex Global Technologies Inc.
789 Market Street, 10th Floor
San Francisco, CA 94103
Attention: Accounts Payable

DESCRIPTION                          QTY    UNIT PRICE     TOTAL
Cloud Server Cluster (c6g.4xlarge)     4       $250.00   $1000.00
Managed PostgreSQL Database (Large)    1       $450.00    $450.00
Data Egress & Backup Storage (TB)      5        $30.00    $150.00
24/7 Enterprise Support Sla            1       $400.00    $400.00

-----------------------------------------------------------------
Subtotal:                                                $2000.00
Tax (8.5%):                                               $170.00
Total Due (USD):                                         $2170.00
-----------------------------------------------------------------
Payment Terms: Net 30. Please remit payment via ACH or Wire."""

SAMPLE_RESUME_FALLBACK = """Jane Doe
Senior Machine Learning Engineer
Seattle, Washington | jane.doe@techinnovate.io | (206) 555-0199 | linkedin.com/in/janedoe-ml

PROFESSIONAL SUMMARY
Experienced Senior Machine Learning Engineer with 7+ years developing production-grade NLP, LLM prompt engineering pipelines, and scalable distributed systems on GCP and AWS. Proven track record leading ML infrastructure teams to reduce latency by 45%.

TECHNICAL SKILLS
- Programming: Python, SQL, C++, Bash
- ML/AI: PyTorch, TensorFlow, Hugging Face, LangChain, Transformers, Pydantic, scikit-learn
- Cloud & Big Data: Google Cloud Platform (BigQuery, Vertex AI, Dataflow), AWS, Docker, Kubernetes
- Tools & Practices: MLOps, CI/CD, Git, REST APIs, FastAPI

WORK EXPERIENCE
Staff Machine Learning Engineer | Nexus AI Labs, Seattle, WA
March 2021 - Present
- Architected and deployed enterprise LLM prompt extraction pipelines serving 2M+ daily requests with 99.9% uptime.
- Optimized fine-tuning and inference pipelines for generative models, cutting GPU cloud spend by $120,000 annually.
- Mentored a team of 6 engineers on structured data extraction, schema validation, and evaluation workflows.

Senior Data Scientist | DataVanguard Inc., San Jose, CA
July 2018 - February 2021
- Built real-time customer recommendation engines using Spark and BigQuery, boosting user engagement by 22%.
- Designed automated document parsing pipelines using regex and deep learning OCR models.
- Led data governance and lineage initiatives to ensure strict GDPR and CCPA compliance.

EDUCATION
Master of Science in Computer Science (Artificial Intelligence)
University of Washington, Seattle, WA (2018)

Bachelor of Science in Computer Engineering
University of California, Berkeley, CA (2016)

CERTIFICATIONS
- Google Cloud Certified Professional Data Engineer (2023)
- AWS Certified Machine Learning - Specialty (2022)"""


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        doc_type = query.get("type", ["invoice"])[0]

        content = ""
        sample_file = "invoice_sample.txt" if doc_type == "invoice" else "resume_sample.txt"
        file_path = os.path.join(root_dir, "samples", sample_file)

        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                pass

        if not content:
            content = SAMPLE_INVOICE_FALLBACK if doc_type == "invoice" else SAMPLE_RESUME_FALLBACK

        self.send_json_response({"type": doc_type, "content": content})

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
