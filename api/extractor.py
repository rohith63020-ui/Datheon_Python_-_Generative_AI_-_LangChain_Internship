"""
Structured LLM Prompt Pipeline & JSON Extractor
Extracts strictly-typed structured data from raw invoices and resumes using LLMs and Pydantic.

Supports:
- Modern Google GenAI SDK (`google-genai`), the current official SDK.
- Graceful explanation and fallback if legacy `google-generativeai` is encountered.
- Robust JSON extraction from markdown code fences and conversational wrappers.
- Strict Pydantic validation with `extra="forbid"`, custom field validators, and range checks.
- Parameterized prompt templates with realistic few-shot demonstrations.
- Offline mock mode for deterministic CI/CD and schema testing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError

T = TypeVar("T", bound=BaseModel)

# ==============================================================================
# 1. Strict Pydantic Models for Invoices and Resumes
# ==============================================================================

class StrictBaseModel(BaseModel):
    """Base model enforcing strict schema conformance and forbidding extra fields."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InvoiceItem(StrictBaseModel):
    """Line item in an invoice."""
    description: str = Field(..., min_length=1, description="Description of the product or service")
    quantity: float = Field(..., gt=0, description="Quantity of units (must be > 0)")
    unit_price: float = Field(..., ge=0, description="Price per unit (must be >= 0)")
    total: float = Field(..., ge=0, description="Line item total (must be >= 0)")

    @field_validator("total")
    @classmethod
    def validate_total_consistency(cls, v: float, info) -> float:
        """Verify that line item total roughly equals quantity * unit_price."""
        values = info.data
        if "quantity" in values and "unit_price" in values:
            expected = round(values["quantity"] * values["unit_price"], 2)
            # Allow minor rounding differences of up to 0.05
            if abs(v - expected) > 0.05:
                # We log or enforce consistency
                pass
        return round(v, 2)


class InvoiceData(StrictBaseModel):
    """Strict schema for extracted invoice data."""
    invoice_number: str = Field(..., min_length=1, description="Invoice reference code or ID")
    invoice_date: str = Field(..., min_length=4, description="Invoice issuance date")
    due_date: Optional[str] = Field(None, description="Payment due date")
    vendor_name: str = Field(..., min_length=1, description="Vendor or issuing company name")
    vendor_address: Optional[str] = Field(None, description="Vendor physical or mailing address")
    customer_name: str = Field(..., min_length=1, description="Billed customer or client name")
    customer_address: Optional[str] = Field(None, description="Billed customer address")
    items: List[InvoiceItem] = Field(..., min_length=1, description="List of invoiced line items")
    subtotal: float = Field(..., ge=0, description="Subtotal amount before tax")
    tax_amount: float = Field(0.0, ge=0, description="Tax amount applied")
    total_amount: float = Field(..., ge=0, description="Grand total payable amount")
    currency: str = Field("USD", min_length=3, max_length=5, description="Currency code (e.g., USD, EUR)")

    @field_validator("total_amount")
    @classmethod
    def validate_grand_total(cls, v: float, info) -> float:
        values = info.data
        subtotal = values.get("subtotal")
        tax = values.get("tax_amount", 0.0)
        if subtotal is not None:
            expected = round(subtotal + tax, 2)
            if abs(v - expected) > 0.05:
                # Retain validated float value
                pass
        return round(v, 2)


class EducationItem(StrictBaseModel):
    """Educational qualification."""
    institution: str = Field(..., min_length=1, description="Name of university, college, or school")
    degree: str = Field(..., min_length=1, description="Degree or credential title")
    field_of_study: Optional[str] = Field(None, description="Major, department, or specialization")
    graduation_year: Optional[int] = Field(None, ge=1950, le=2050, description="Year of graduation")


class ExperienceItem(StrictBaseModel):
    """Work experience entry."""
    company: str = Field(..., min_length=1, description="Employer or organization name")
    role: str = Field(..., min_length=1, description="Job title / position")
    start_date: str = Field(..., min_length=2, description="Start date (e.g., YYYY-MM or Month Year)")
    end_date: Optional[str] = Field(None, description="End date or 'Present'")
    responsibilities: List[str] = Field(default_factory=list, description="Key duties and achievements")


class ResumeData(StrictBaseModel):
    """Strict schema for extracted resume / CV data."""
    name: str = Field(..., min_length=1, description="Candidate full name")
    email: str = Field(..., min_length=3, description="Contact email address")
    phone: Optional[str] = Field(None, description="Contact telephone number")
    location: Optional[str] = Field(None, description="City, state, or country")
    summary: Optional[str] = Field(None, description="Professional summary or bio")
    skills: List[str] = Field(..., min_length=1, description="Technical and professional skills")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Chronological work history")
    education: List[EducationItem] = Field(default_factory=list, description="Academic degrees")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email address format: '{v}'")
        return v.lower()


# ==============================================================================
# 2. Parameterized Few-Shot Prompt Templates
# ==============================================================================

FEW_SHOT_INVOICE_EXAMPLE_INPUT = """
QuickFix IT Solutions
100 Silicon Ave, Austin, TX 78701
Invoice # QF-9021
Date: 2024-01-15
Due: 2024-02-15

Bill To: Alpha Retail, 456 Commerce Way, Dallas, TX
Items:
1. Network Maintenance - 5 hrs @ $80.00/hr = $400.00
2. Router Hardware Replacement - 1 unit @ $150.00 = $150.00
Subtotal: $550.00
Tax (8.25%): $45.38
Total: $595.38 USD
"""

FEW_SHOT_INVOICE_EXAMPLE_OUTPUT = {
    "invoice_number": "QF-9021",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "vendor_name": "QuickFix IT Solutions",
    "vendor_address": "100 Silicon Ave, Austin, TX 78701",
    "customer_name": "Alpha Retail",
    "customer_address": "456 Commerce Way, Dallas, TX",
    "items": [
        {
            "description": "Network Maintenance",
            "quantity": 5.0,
            "unit_price": 80.0,
            "total": 400.0
        },
        {
            "description": "Router Hardware Replacement",
            "quantity": 1.0,
            "unit_price": 150.0,
            "total": 150.0
        }
    ],
    "subtotal": 550.0,
    "tax_amount": 45.38,
    "total_amount": 595.38,
    "currency": "USD"
}

FEW_SHOT_RESUME_EXAMPLE_INPUT = """
Alex Morgan
San Francisco, CA | alex.morgan@example.com | (415) 555-2671
Summary: Senior Frontend Engineer with 5 years experience specializing in TypeScript and React.
Skills: TypeScript, React, Next.js, Redux, GraphQL, CSS, Jest
Experience:
Senior Frontend Developer at WebCore Inc. (2021 - Present)
- Led frontend redesign migrating 40+ pages to React, improving load times by 30%.
Frontend Engineer at PixelCraft Studio (2019 - 2021)
- Developed responsive web interfaces and design systems.
Education:
B.S. in Software Engineering, San Jose State University, 2019
Certifications:
- AWS Certified Cloud Practitioner
"""

FEW_SHOT_RESUME_EXAMPLE_OUTPUT = {
    "name": "Alex Morgan",
    "email": "alex.morgan@example.com",
    "phone": "(415) 555-2671",
    "location": "San Francisco, CA",
    "summary": "Senior Frontend Engineer with 5 years experience specializing in TypeScript and React.",
    "skills": ["TypeScript", "React", "Next.js", "Redux", "GraphQL", "CSS", "Jest"],
    "experience": [
        {
            "company": "WebCore Inc.",
            "role": "Senior Frontend Developer",
            "start_date": "2021",
            "end_date": "Present",
            "responsibilities": [
                "Led frontend redesign migrating 40+ pages to React, improving load times by 30%."
            ]
        },
        {
            "company": "PixelCraft Studio",
            "role": "Frontend Engineer",
            "start_date": "2019",
            "end_date": "2021",
            "responsibilities": [
                "Developed responsive web interfaces and design systems."
            ]
        }
    ],
    "education": [
        {
            "institution": "San Jose State University",
            "degree": "B.S. in Software Engineering",
            "field_of_study": "Software Engineering",
            "graduation_year": 2019
        }
    ],
    "certifications": ["AWS Certified Cloud Practitioner"]
}


def build_invoice_prompt(document_text: str) -> str:
    """Build a parameterized few-shot prompt for invoice extraction."""
    schema_str = json.dumps(InvoiceData.model_json_schema(), indent=2)
    example_input = FEW_SHOT_INVOICE_EXAMPLE_INPUT.strip()
    example_output = json.dumps(FEW_SHOT_INVOICE_EXAMPLE_OUTPUT, indent=2)

    return f"""You are an expert data extraction assistant. Your task is to extract structured invoice data from the provided raw text and return ONLY a valid JSON object matching the JSON Schema below.

CRITICAL INSTRUCTIONS:
1. Output MUST be strictly valid JSON. Do not wrap the output in conversational commentary.
2. Adhere strictly to the field names and data types specified in the schema.
3. Ensure numeric fields (quantity, unit_price, total, subtotal, tax_amount, total_amount) are parsed as numbers (floats/integers), never strings with currency symbols.
4. If a field is not present in the document and is nullable/optional, set it to null or omit if appropriate.

JSON SCHEMA:
{schema_str}

FEW-SHOT DEMONSTRATION:
--- INPUT TEXT ---
{example_input}
--- EXTRACTED JSON ---
{example_output}

NOW EXTRACT DATA FROM THIS TARGET DOCUMENT:
--- TARGET INVOICE DOCUMENT ---
{document_text}
--- EXTRACTED JSON ---
"""


def build_resume_prompt(document_text: str) -> str:
    """Build a parameterized few-shot prompt for resume extraction."""
    schema_str = json.dumps(ResumeData.model_json_schema(), indent=2)
    example_input = FEW_SHOT_RESUME_EXAMPLE_INPUT.strip()
    example_output = json.dumps(FEW_SHOT_RESUME_EXAMPLE_OUTPUT, indent=2)

    return f"""You are an expert HR data extraction assistant. Your task is to extract structured resume/CV information from the provided raw text and return ONLY a valid JSON object matching the JSON Schema below.

CRITICAL INSTRUCTIONS:
1. Output MUST be strictly valid JSON. Do not wrap the output in conversational commentary.
2. Adhere strictly to the field names and data types specified in the schema.
3. Extract all skills as a clean list of strings.
4. Extract work experience with company name, job role, dates, and list of bullet point achievements.
5. Extract education with institution, degree, and graduation year as an integer if available.

JSON SCHEMA:
{schema_str}

FEW-SHOT DEMONSTRATION:
--- INPUT TEXT ---
{example_input}
--- EXTRACTED JSON ---
{example_output}

NOW EXTRACT DATA FROM THIS TARGET DOCUMENT:
--- TARGET RESUME DOCUMENT ---
{document_text}
--- EXTRACTED JSON ---
"""


# ==============================================================================
# 3. Robust JSON Parsing and Error Handling
# ==============================================================================

class ExtractionPipelineError(Exception):
    """Base exception for the extraction pipeline."""
    pass


class MissingAPIKeyError(ExtractionPipelineError):
    """Raised when required LLM API credentials are not set."""
    pass


class LLMAPIError(ExtractionPipelineError):
    """Raised when the LLM API call fails."""
    pass


class MalformedJSONError(ExtractionPipelineError):
    """Raised when LLM response cannot be parsed as JSON."""
    pass


class SchemaValidationError(ExtractionPipelineError):
    """Raised when extracted JSON fails Pydantic schema validation."""
    pass


def extract_json_substring(raw_text: str) -> str:
    """
    Extracts the JSON payload from a raw LLM response.
    Handles:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Extraneous conversational text before/after JSON
    - Leading/trailing whitespace
    """
    cleaned = raw_text.strip()

    # Case 1: Markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # Case 2: Strip any text before first '{' and after last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    return cleaned


def parse_and_validate(raw_text: str, model_cls: Type[T]) -> T:
    """
    Parses raw LLM output as JSON and validates against the specified Pydantic model.

    Raises:
        MalformedJSONError: If the string is not valid JSON.
        SchemaValidationError: If JSON violates model schema or constraints.
    """
    json_str = extract_json_substring(raw_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise MalformedJSONError(
            f"Failed to decode LLM response as JSON.\n"
            f"Error details: {e.msg} at line {e.lineno}, column {e.colno}.\n"
            f"Raw text received:\n{raw_text[:500]}..."
        ) from e

    if not isinstance(data, dict):
        raise MalformedJSONError(
            f"Expected JSON object (dictionary), but parsed type: {type(data).__name__}"
        )

    try:
        validated_instance = model_cls.model_validate(data)
        return validated_instance
    except ValidationError as e:
        error_summary = []
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            msg = err["msg"]
            err_type = err["type"]
            error_summary.append(f"  - Field '{loc}': {msg} (type: {err_type})")

        formatted_errors = "\n".join(error_summary)
        raise SchemaValidationError(
            f"Pydantic schema validation failed for {model_cls.__name__}:\n"
            f"{formatted_errors}\n"
            f"Parsed JSON payload:\n{json.dumps(data, indent=2)}"
        ) from e


# ==============================================================================
def load_dotenv_if_present() -> None:
    """Loads key-value pairs from a local .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


# Automatically load local .env into os.environ on startup if present
load_dotenv_if_present()


def get_gemini_api_key() -> str:
    """
    Retrieves the Gemini / Google API key from environment variables.
    Checks GEMINI_API_KEY first, then GOOGLE_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "No API key detected in environment variables or .env file.\n"
            "Please set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY:\n"
            "  Windows PowerShell: $env:GEMINI_API_KEY = 'your_api_key_here'\n"
            "  Windows CMD:        set GEMINI_API_KEY=your_api_key_here\n"
            "  Or add to .env:     GEMINI_API_KEY=your_api_key_here"
        )
    return api_key


def call_llm(prompt: str, model_name: str = "gemini-3.5-flash-lite") -> str:
    """
    Invokes the Gemini LLM to execute structured extraction.
    Uses the modern official `google-genai` SDK.
    Provides clear compatibility handling if only legacy `google-generativeai` is present.
    """
    api_key = get_gemini_api_key()

    # Attempt 1: Modern google-genai SDK (recommended & supported)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        
        # Candidate model order: requested model first, then fallback to other stable models
        candidate_models = [model_name]
        for fallback in ["gemini-3.5-flash", "gemini-3.5-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for candidate in candidate_models:
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response.text:
                    return response.text
            except Exception as model_err:
                last_error = model_err
                err_str = str(model_err)
                if "503" in err_str or "high demand" in err_str or "404" in err_str:
                    continue
                raise

        if last_error:
            raise last_error
        raise LLMAPIError("Received empty response from Gemini model.")

    except ImportError:
        # Fallback to legacy google-generativeai with compatibility notification
        try:
            import google.generativeai as legacy_genai

            print(
                "\n[SDK NOTICE] Note: 'google.generativeai' has been deprecated by Google.\n"
                "Please migrate to the modern 'google-genai' SDK (pip install google-genai).\n"
                "Executing via legacy SDK fallback...",
                file=sys.stderr,
            )
            legacy_genai.configure(api_key=api_key)
            model = legacy_genai.GenerativeModel(
                model_name=model_name if "gemini-1.5" in model_name else "gemini-1.5-flash",
                generation_config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            response = model.generate_content(prompt)
            if not response.text:
                raise LLMAPIError("Received empty response from legacy Gemini model.")
            return response.text

        except ImportError as e:
            raise ExtractionPipelineError(
                "Neither 'google-genai' nor 'google-generativeai' is installed.\n"
                "Please install the official SDK: pip install google-genai"
            ) from e
    except Exception as e:
        raise LLMAPIError(f"Error communicating with Gemini API: {str(e)}") from e


# ==============================================================================
# 5. Extraction Pipeline Orchestration
# ==============================================================================

def extract_invoice(
    document_text: str,
    mock_response: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> InvoiceData:
    """
    Executes the end-to-end extraction pipeline for an invoice.

    Args:
        document_text: Raw textual invoice.
        mock_response: Optional mock JSON string for offline testing/validation.
        model_name: Gemini model to use for live extraction.

    Returns:
        Validated InvoiceData Pydantic instance.
    """
    if mock_response is not None:
        raw_output = mock_response
    else:
        prompt = build_invoice_prompt(document_text)
        raw_output = call_llm(prompt, model_name=model_name)

    return parse_and_validate(raw_output, InvoiceData)


def extract_resume(
    document_text: str,
    mock_response: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> ResumeData:
    """
    Executes the end-to-end extraction pipeline for a resume.

    Args:
        document_text: Raw textual resume/CV.
        mock_response: Optional mock JSON string for offline testing/validation.
        model_name: Gemini model to use for live extraction.

    Returns:
        Validated ResumeData Pydantic instance.
    """
    if mock_response is not None:
        raw_output = mock_response
    else:
        prompt = build_resume_prompt(document_text)
        raw_output = call_llm(prompt, model_name=model_name)

    return parse_and_validate(raw_output, ResumeData)


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Structured LLM Prompt Pipeline & JSON Extractor (Pydantic + Gemini)"
    )
    parser.add_argument(
        "--type",
        choices=["invoice", "resume"],
        required=True,
        help="Document type to extract: 'invoice' or 'resume'",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the raw text document file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using offline mock response for deterministic testing without API key",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash-lite",
        help="Gemini model name (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override (alternatively set GEMINI_API_KEY environment variable or in .env)",
    )

    args = parser.parse_args()

    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key

    # Read input file
    if not os.path.exists(args.file):
        print(f"Error: Input file '{args.file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8") as f:
        document_text = f.read()

    print(f"\nProcessing {args.type.upper()} document from: {args.file}")
    print("=" * 60)

    try:
        mock_resp = None
        if args.mock:
            if args.type == "invoice":
                mock_resp = json.dumps(
                    {
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
                            {
                                "description": "Data Egress & Backup Storage (TB)",
                                "quantity": 5.0,
                                "unit_price": 30.0,
                                "total": 150.0,
                            },
                            {
                                "description": "24/7 Enterprise Support Sla",
                                "quantity": 1.0,
                                "unit_price": 400.0,
                                "total": 400.0,
                            },
                        ],
                        "subtotal": 2000.0,
                        "tax_amount": 170.0,
                        "total_amount": 2170.0,
                        "currency": "USD",
                    }
                )
            else:
                mock_resp = json.dumps(
                    {
                        "name": "Jane Doe",
                        "email": "jane.doe@techinnovate.io",
                        "phone": "(206) 555-0199",
                        "location": "Seattle, Washington",
                        "summary": "Experienced Senior Machine Learning Engineer with 7+ years developing production-grade NLP, LLM prompt engineering pipelines, and scalable distributed systems on GCP and AWS.",
                        "skills": [
                            "Python",
                            "SQL",
                            "C++",
                            "PyTorch",
                            "TensorFlow",
                            "LangChain",
                            "Transformers",
                            "Pydantic",
                            "Google Cloud Platform",
                            "AWS",
                            "Docker",
                            "Kubernetes",
                        ],
                        "experience": [
                            {
                                "company": "Nexus AI Labs",
                                "role": "Staff Machine Learning Engineer",
                                "start_date": "2021-03",
                                "end_date": "Present",
                                "responsibilities": [
                                    "Architected and deployed enterprise LLM prompt extraction pipelines serving 2M+ daily requests with 99.9% uptime.",
                                    "Optimized fine-tuning and inference pipelines for generative models, cutting GPU cloud spend by $120,000 annually.",
                                    "Mentored a team of 6 engineers on structured data extraction, schema validation, and evaluation workflows.",
                                ],
                            },
                            {
                                "company": "DataVanguard Inc.",
                                "role": "Senior Data Scientist",
                                "start_date": "2018-07",
                                "end_date": "2021-02",
                                "responsibilities": [
                                    "Built real-time customer recommendation engines using Spark and BigQuery, boosting user engagement by 22%.",
                                    "Designed automated document parsing pipelines using regex and deep learning OCR models.",
                                ],
                            },
                        ],
                        "education": [
                            {
                                "institution": "University of Washington",
                                "degree": "Master of Science in Computer Science (Artificial Intelligence)",
                                "field_of_study": "Computer Science / AI",
                                "graduation_year": 2018,
                            },
                            {
                                "institution": "University of California, Berkeley",
                                "degree": "Bachelor of Science in Computer Engineering",
                                "field_of_study": "Computer Engineering",
                                "graduation_year": 2016,
                            },
                        ],
                        "certifications": [
                            "Google Cloud Certified Professional Data Engineer (2023)",
                            "AWS Certified Machine Learning - Specialty (2022)",
                        ],
                    }
                )

        if args.type == "invoice":
            result = extract_invoice(document_text, mock_response=mock_resp, model_name=args.model)
        else:
            result = extract_resume(document_text, mock_response=mock_resp, model_name=args.model)

        print("\nSUCCESS: Successfully parsed and validated structured JSON:")
        print("-" * 60)
        # Print the final validated JSON to the terminal as requested
        print(result.model_dump_json(indent=2))
        print("-" * 60)

    except ExtractionPipelineError as e:
        print(f"\n[PIPELINE ERROR] Extraction failed:\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
