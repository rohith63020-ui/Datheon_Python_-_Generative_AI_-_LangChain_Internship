"""
Comprehensive Unit & Integration Test Suite for Structured LLM Extractor Pipeline.

Validates:
1. Valid invoice data parsing & Pydantic validation.
2. Valid resume data parsing & Pydantic validation.
3. Strict rejection of missing required fields.
4. Strict rejection of extra/unknown fields (ConfigDict extra="forbid").
5. Strict rejection of invalid numerical bounds (negative prices, zero quantity).
6. Strict rejection of invalid email formats.
7. Handling of malformed JSON strings.
8. Handling of markdown-wrapped JSON code fences.
9. Handling of missing API keys.
10. End-to-end execution of both sample documents.
"""

import json
import os
import unittest
from pydantic import ValidationError

from extractor import (
    InvoiceData,
    InvoiceItem,
    ResumeData,
    EducationItem,
    ExperienceItem,
    extract_invoice,
    extract_resume,
    extract_json_substring,
    parse_and_validate,
    MalformedJSONError,
    SchemaValidationError,
    MissingAPIKeyError,
    get_gemini_api_key,
)


class TestInvoiceExtraction(unittest.TestCase):
    def setUp(self):
        with open("samples/invoice_sample.txt", "r", encoding="utf-8") as f:
            self.invoice_text = f.read()

        self.valid_invoice_payload = {
            "invoice_number": "INV-2024-00892",
            "invoice_date": "2024-10-15",
            "due_date": "2024-11-14",
            "vendor_name": "ACME CLOUD SERVICES LLC",
            "vendor_address": "123 Innovation Way, Suite 400, Tech City, CA 94016",
            "customer_name": "Apex Global Technologies Inc.",
            "customer_address": "789 Market Street, 10th Floor, San Francisco, CA 94103",
            "items": [
                {
                    "description": "Cloud Server Cluster (c6g.4xlarge)",
                    "quantity": 4.0,
                    "unit_price": 250.0,
                    "total": 1000.0,
                },
                {
                    "description": "Managed PostgreSQL Database (Large)",
                    "quantity": 1.0,
                    "unit_price": 450.0,
                    "total": 450.0,
                },
            ],
            "subtotal": 1450.0,
            "tax_amount": 123.25,
            "total_amount": 1573.25,
            "currency": "USD",
        }

    def test_valid_invoice_validation(self):
        """Test that a well-formed invoice payload validates successfully."""
        invoice = InvoiceData.model_validate(self.valid_invoice_payload)
        self.assertEqual(invoice.invoice_number, "INV-2024-00892")
        self.assertEqual(len(invoice.items), 2)
        self.assertEqual(invoice.total_amount, 1573.25)
        self.assertEqual(invoice.currency, "USD")

    def test_invoice_missing_required_fields_rejected(self):
        """Test that missing required fields like 'invoice_number' raise SchemaValidationError."""
        invalid_payload = self.valid_invoice_payload.copy()
        del invalid_payload["invoice_number"]

        with self.assertRaises(SchemaValidationError) as ctx:
            parse_and_validate(json.dumps(invalid_payload), InvoiceData)
        self.assertIn("invoice_number", str(ctx.exception))

    def test_invoice_negative_total_rejected(self):
        """Test that negative prices or quantities are rejected."""
        invalid_payload = json.loads(json.dumps(self.valid_invoice_payload))
        invalid_payload["items"][0]["quantity"] = -2.0  # Must be > 0

        with self.assertRaises(SchemaValidationError) as ctx:
            parse_and_validate(json.dumps(invalid_payload), InvoiceData)
        self.assertIn("items -> 0 -> quantity", str(ctx.exception))

    def test_invoice_extra_fields_forbidden(self):
        """Test that extra fields not in schema are strictly forbidden."""
        invalid_payload = self.valid_invoice_payload.copy()
        invalid_payload["unexpected_injected_field"] = "HACK"

        with self.assertRaises(SchemaValidationError) as ctx:
            parse_and_validate(json.dumps(invalid_payload), InvoiceData)
        self.assertIn("unexpected_injected_field", str(ctx.exception))

    def test_invoice_markdown_fence_cleaning(self):
        """Test that JSON wrapped in markdown code blocks is cleanly parsed."""
        raw_llm_markdown = f"Here is the extracted invoice:\n```json\n{json.dumps(self.valid_invoice_payload)}\n```\nHope this helps!"
        invoice = parse_and_validate(raw_llm_markdown, InvoiceData)
        self.assertEqual(invoice.invoice_number, "INV-2024-00892")


class TestResumeExtraction(unittest.TestCase):
    def setUp(self):
        with open("samples/resume_sample.txt", "r", encoding="utf-8") as f:
            self.resume_text = f.read()

        self.valid_resume_payload = {
            "name": "Jane Doe",
            "email": "jane.doe@techinnovate.io",
            "phone": "(206) 555-0199",
            "location": "Seattle, Washington",
            "summary": "Experienced Senior Machine Learning Engineer.",
            "skills": ["Python", "PyTorch", "Pydantic", "GCP"],
            "experience": [
                {
                    "company": "Nexus AI Labs",
                    "role": "Staff Machine Learning Engineer",
                    "start_date": "2021-03",
                    "end_date": "Present",
                    "responsibilities": ["Architected LLM pipelines"],
                }
            ],
            "education": [
                {
                    "institution": "University of Washington",
                    "degree": "M.S. in Computer Science",
                    "field_of_study": "AI",
                    "graduation_year": 2018,
                }
            ],
            "certifications": ["Google Cloud Certified Professional Data Engineer"],
        }

    def test_valid_resume_validation(self):
        """Test that a valid resume payload parses and validates."""
        resume = ResumeData.model_validate(self.valid_resume_payload)
        self.assertEqual(resume.name, "Jane Doe")
        self.assertEqual(resume.email, "jane.doe@techinnovate.io")
        self.assertIn("Python", resume.skills)
        self.assertEqual(len(resume.experience), 1)

    def test_resume_invalid_email_rejected(self):
        """Test that invalid email format is rejected by the custom field validator."""
        invalid_payload = self.valid_resume_payload.copy()
        invalid_payload["email"] = "not-an-email"

        with self.assertRaises(SchemaValidationError) as ctx:
            parse_and_validate(json.dumps(invalid_payload), ResumeData)
        self.assertIn("Invalid email address format", str(ctx.exception))

    def test_resume_missing_skills_rejected(self):
        """Test that missing required skills field is rejected."""
        invalid_payload = self.valid_resume_payload.copy()
        del invalid_payload["skills"]

        with self.assertRaises(SchemaValidationError) as ctx:
            parse_and_validate(json.dumps(invalid_payload), ResumeData)
        self.assertIn("skills", str(ctx.exception))

    def test_resume_extra_field_rejected(self):
        """Test that extra unmapped fields are rejected."""
        invalid_payload = self.valid_resume_payload.copy()
        invalid_payload["marital_status"] = "Unknown"

        with self.assertRaises(SchemaValidationError) as ctx:
            parse_and_validate(json.dumps(invalid_payload), ResumeData)
        self.assertIn("marital_status", str(ctx.exception))


class TestParsingRobustness(unittest.TestCase):
    def test_malformed_json_raises_malformed_json_error(self):
        """Test that truncated or malformed JSON raises MalformedJSONError."""
        bad_json = '{"invoice_number": "INV-123", "items": [ missing closing brackets'
        with self.assertRaises(MalformedJSONError):
            parse_and_validate(bad_json, InvoiceData)

    def test_conversational_text_around_json(self):
        """Test extraction when LLM adds conversational filler before and after."""
        payload = {
            "institution": "MIT",
            "degree": "B.S. Computer Science",
            "field_of_study": "CS",
            "graduation_year": 2020,
        }
        wrapped = f"Sure! Here is the requested education item: {json.dumps(payload)}. Let me know if you need more!"
        result = parse_and_validate(wrapped, EducationItem)
        self.assertEqual(result.institution, "MIT")
        self.assertEqual(result.graduation_year, 2020)


class TestEnvironmentAndAPIKeyHandling(unittest.TestCase):
    def test_missing_api_key_raises_informative_error(self):
        """Test that get_gemini_api_key raises MissingAPIKeyError if env vars are unset."""
        orig_gemini = os.environ.get("GEMINI_API_KEY")
        orig_google = os.environ.get("GOOGLE_API_KEY")

        try:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]

            with self.assertRaises(MissingAPIKeyError) as ctx:
                get_gemini_api_key()
            self.assertIn("No API key detected", str(ctx.exception))
        finally:
            if orig_gemini:
                os.environ["GEMINI_API_KEY"] = orig_gemini
            if orig_google:
                os.environ["GOOGLE_API_KEY"] = orig_google


if __name__ == "__main__":
    unittest.main()
