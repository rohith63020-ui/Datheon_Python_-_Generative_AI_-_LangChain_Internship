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


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        doc_type = query.get("type", ["invoice"])[0]

        sample_file = "invoice_sample.txt" if doc_type == "invoice" else "resume_sample.txt"
        file_path = os.path.join(root_dir, "samples", sample_file)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.send_json_response({"type": doc_type, "content": content})
        else:
            self.send_json_response({"error": "Sample file not found"}, status=404)

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
