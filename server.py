"""
Local Web Application Server for Structured LLM Extractor & JSON Pipeline.
Serves a modern, interactive web UI at http://localhost:8000 with live extraction,
Pydantic schema validation, visual data inspection, and error diagnostics.
Zero extra dependencies - uses Python's built-in http.server.
"""

from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from extractor import (
    extract_invoice,
    extract_resume,
    ExtractionPipelineError,
    MalformedJSONError,
    SchemaValidationError,
    MissingAPIKeyError,
    LLMAPIError,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8000


class ExtractorHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handles static web requests and JSON extraction API endpoints."""

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            index_path = os.path.join(BASE_DIR, "index.html")
            if not os.path.exists(index_path):
                index_path = os.path.join(BASE_DIR, "public", "index.html")
            self.serve_file(index_path, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/sample":
            query = parse_qs(parsed.query)
            doc_type = query.get("type", ["invoice"])[0]
            sample_file = "invoice_sample.txt" if doc_type == "invoice" else "resume_sample.txt"
            path = os.path.join(BASE_DIR, "samples", sample_file)

            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_json_response({"type": doc_type, "content": content})
            else:
                self.send_json_response({"error": "Sample file not found"}, status=404)
            return

        # Check for static file in public or base dir
        clean_path = parsed.path.lstrip("/")
        candidate_paths = [
            os.path.join(BASE_DIR, "public", clean_path),
            os.path.join(BASE_DIR, clean_path),
        ]
        for cp in candidate_paths:
            if os.path.isfile(cp):
                content_type = "text/plain"
                if cp.endswith(".html"):
                    content_type = "text/html; charset=utf-8"
                elif cp.endswith(".css"):
                    content_type = "text/css"
                elif cp.endswith(".js"):
                    content_type = "application/javascript"
                elif cp.endswith(".json"):
                    content_type = "application/json"
                self.serve_file(cp, content_type)
                return

        self.send_error(HTTPStatus.NOT_FOUND, "File not found")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/extract":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self.send_json_response(
                    {"status": "error", "error_type": "JSONDecodeError", "message": "Invalid request payload."},
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
                if "CORRUPT" in doc_text.upper():
                    mock_response = json.dumps({
                        "invoice_number": "INV-FAIL-01",
                        "invoice_date": "2024-10-15",
                        "vendor_name": "Glitch Systems",
                        "customer_name": "Bad Customer",
                        "items": [
                            {"description": "Corrupted Storage", "quantity": -10.0, "unit_price": 50.0, "total": -500.0}
                        ],
                        "subtotal": 500.0,
                        "tax_amount": 0.0,
                        "total_amount": 500.0,
                        "currency": "USD"
                    })
                elif doc_type == "invoice":
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
            return

        self.send_json_response({"error": "Endpoint not found"}, status=404)

    def serve_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Clean logging format
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run_server(port: int = PORT):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, ExtractorHTTPRequestHandler)
    print(f"\n=======================================================")
    print(f"  Structured LLM Extractor Web App is LIVE!")
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"  Local address:        http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


# Export top-level entrypoints recognized by Vercel Serverless Python runtime
handler = ExtractorHTTPRequestHandler
app = ExtractorHTTPRequestHandler
application = ExtractorHTTPRequestHandler


if __name__ == "__main__":
    run_server()
