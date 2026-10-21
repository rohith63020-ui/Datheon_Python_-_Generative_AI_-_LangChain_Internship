"""
Vercel Serverless Function: /api/extract
Handles structured data extraction for invoices and resumes.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Add root directory to sys.path so extractor can be imported cleanly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from extractor import (
    extract_invoice,
    extract_resume,
    SchemaValidationError,
    MalformedJSONError,
    MissingAPIKeyError,
    LLMAPIError,
)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_json_response(
                {"status": "error", "error_type": "JSONDecodeError", "message": "Invalid JSON in request body."},
                status=400,
            )
            return

        doc_type = payload.get("type", "invoice")
        doc_text = payload.get("text", "").strip()
        use_mock = bool(payload.get("mock", False))
        custom_api_key = payload.get("api_key", "").strip()

        if not doc_text:
            self.send_json_response(
                {"status": "error", "error_type": "EmptyInputError", "message": "Document text cannot be empty."},
                status=400,
            )
            return

        if custom_api_key:
            os.environ["GEMINI_API_KEY"] = custom_api_key

        mock_response = None
        if use_mock:
            if doc_type == "invoice":
                mock_response = json.dumps({
                    "invoice_number": "INV-2024-00892",
                    "invoice_date": "2024-10-15",
                    "due_date": "2024-11-14",
                    "vendor_name": "ACME CLOUD SERVICES LLC",
                    "vendor_address": "123 Innovation Way, Suite 400, Tech City, CA 94016",
                    "customer_name": "Apex Global Technologies Inc.",
                    "customer_address": "789 Market Street, 10th Floor, San Francisco, CA 94103",
                    "items": [
                        {"description": "Cloud Server Cluster (c6g.4xlarge)", "quantity": 4.0, "unit_price": 250.0, "total": 1000.0},
                        {"description": "Managed PostgreSQL Database (Large)", "quantity": 1.0, "unit_price": 450.0, "total": 450.0},
                        {"description": "Data Egress & Backup Storage (TB)", "quantity": 5.0, "unit_price": 30.0, "total": 150.0},
                        {"description": "24/7 Enterprise Support Sla", "quantity": 1.0, "unit_price": 400.0, "total": 400.0}
                    ],
                    "subtotal": 2000.0,
                    "tax_amount": 170.0,
                    "total_amount": 2170.0,
                    "currency": "USD"
                })
            else:
                mock_response = json.dumps({
                    "name": "Jane Doe",
                    "email": "jane.doe@techinnovate.io",
                    "phone": "(206) 555-0199",
                    "location": "Seattle, Washington",
                    "summary": "Experienced Senior Machine Learning Engineer with 7+ years developing production-grade NLP, LLM prompt engineering pipelines, and scalable distributed systems on GCP and AWS.",
                    "skills": ["Python", "SQL", "C++", "PyTorch", "TensorFlow", "LangChain", "Transformers", "Pydantic", "Google Cloud Platform", "AWS", "Docker", "Kubernetes"],
                    "experience": [
                        {
                            "company": "Nexus AI Labs",
                            "role": "Staff Machine Learning Engineer",
                            "start_date": "March 2021",
                            "end_date": "Present",
                            "responsibilities": [
                                "Architected and deployed enterprise LLM prompt extraction pipelines serving 2M+ daily requests with 99.9% uptime.",
                                "Optimized fine-tuning and inference pipelines for generative models, cutting GPU cloud spend by $120,000 annually.",
                                "Mentored a team of 6 engineers on structured data extraction, schema validation, and evaluation workflows."
                            ]
                        }
                    ],
                    "education": [
                        {"institution": "University of Washington", "degree": "Master of Science in Computer Science", "field_of_study": "Artificial Intelligence", "graduation_year": 2018}
                    ],
                    "certifications": [
                        "Google Cloud Certified Professional Data Engineer (2023)",
                        "AWS Certified Machine Learning - Specialty (2022)"
                    ]
                })

        try:
            if doc_type == "invoice":
                validated_data = extract_invoice(doc_text, mock_response=mock_response)
            else:
                validated_data = extract_resume(doc_text, mock_response=mock_response)

            self.send_json_response({
                "status": "success",
                "type": doc_type,
                "data": validated_data.model_dump(),
                "json_str": validated_data.model_dump_json(indent=2)
            })

        except SchemaValidationError as e:
            self.send_json_response({
                "status": "error",
                "error_type": "SchemaValidationError",
                "message": str(e)
            }, status=422)
        except MalformedJSONError as e:
            self.send_json_response({
                "status": "error",
                "error_type": "MalformedJSONError",
                "message": str(e)
            }, status=400)
        except MissingAPIKeyError as e:
            self.send_json_response({
                "status": "error",
                "error_type": "MissingAPIKeyError",
                "message": str(e)
            }, status=401)
        except LLMAPIError as e:
            self.send_json_response({
                "status": "error",
                "error_type": "LLMAPIError",
                "message": str(e)
            }, status=502)
        except Exception as e:
            self.send_json_response({
                "status": "error",
                "error_type": type(e).__name__,
                "message": f"Unexpected extraction error: {str(e)}"
            }, status=500)

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
