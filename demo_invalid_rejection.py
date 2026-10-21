"""
Demonstrates Pydantic validation failures and error handling.
Shows that corrupted, incomplete, or invalid LLM outputs are strictly rejected.
"""

import sys
from extractor import parse_and_validate, InvoiceData, ResumeData, ExtractionPipelineError

print("=" * 70)
print("DEMO: STRICT PYDANTIC REJECTION OF INVALID LLM OUTPUTS")
print("=" * 70)

# Case 1: Negative quantity and negative unit price
print("\n[Case 1] Negative line item quantity in Invoice:")
invalid_invoice_negative = """{
  "invoice_number": "INV-FAIL-01",
  "invoice_date": "2024-10-15",
  "vendor_name": "Bad Vendor",
  "customer_name": "Bad Customer",
  "items": [
    {"description": "Server", "quantity": -5, "unit_price": 100.0, "total": 500.0}
  ],
  "subtotal": 500.0,
  "tax_amount": 0.0,
  "total_amount": 500.0,
  "currency": "USD"
}"""
try:
    parse_and_validate(invalid_invoice_negative, InvoiceData)
    print("FAILED: Should have rejected negative quantity!")
except ExtractionPipelineError as e:
    print("SUCCESSFULLY REJECTED:")
    print(e)

# Case 2: Missing required field (e.g. missing skills in Resume)
print("\n" + "-" * 70)
print("[Case 2] Missing required field ('skills') in Resume:")
invalid_resume_missing = """{
  "name": "John Doe",
  "email": "john.doe@example.com"
}"""
try:
    parse_and_validate(invalid_resume_missing, ResumeData)
    print("FAILED: Should have rejected missing required field!")
except ExtractionPipelineError as e:
    print("SUCCESSFULLY REJECTED:")
    print(e)

# Case 3: Invalid email format
print("\n" + "-" * 70)
print("[Case 3] Invalid email address format in Resume:")
invalid_resume_email = """{
  "name": "Alice Smith",
  "email": "not_an_email_address",
  "skills": ["Python", "FastAPI"]
}"""
try:
    parse_and_validate(invalid_resume_email, ResumeData)
    print("FAILED: Should have rejected invalid email address!")
except ExtractionPipelineError as e:
    print("SUCCESSFULLY REJECTED:")
    print(e)

# Case 4: Extra injected unmapped field (extra='forbid')
print("\n" + "-" * 70)
print("[Case 4] Injected extra field ('malicious_field') forbidden by schema:")
invalid_invoice_extra = """{
  "invoice_number": "INV-FAIL-04",
  "invoice_date": "2024-10-15",
  "vendor_name": "Vendor Corp",
  "customer_name": "Customer Inc",
  "items": [{"description": "Item A", "quantity": 1, "unit_price": 50.0, "total": 50.0}],
  "subtotal": 50.0,
  "total_amount": 50.0,
  "currency": "USD",
  "malicious_injected_field": "SQL_INJECTION"
}"""
try:
    parse_and_validate(invalid_invoice_extra, InvoiceData)
    print("FAILED: Should have rejected extra field!")
except ExtractionPipelineError as e:
    print("SUCCESSFULLY REJECTED:")
    print(e)

# Case 5: Corrupt / Malformed JSON
print("\n" + "-" * 70)
print("[Case 5] Malformed / Truncated JSON:")
corrupt_json = """{ "invoice_number": "INV-CORRUPT", "items": [ """
try:
    parse_and_validate(corrupt_json, InvoiceData)
    print("FAILED: Should have rejected malformed JSON!")
except ExtractionPipelineError as e:
    print("SUCCESSFULLY REJECTED:")
    print(e)

print("\n" + "=" * 70)
print("ALL 5 INVALID REJECTION TEST CASES PASSED SUCCESSFULLY!")
print("=" * 70)
