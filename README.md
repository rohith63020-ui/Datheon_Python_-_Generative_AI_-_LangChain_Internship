# Structured LLM Prompt Pipeline & JSON Extractor

A production-grade Python extraction pipeline that extracts strictly-typed, validated JSON from unstructured document text (invoices and resumes) using Gemini LLMs and Pydantic.

---

## 1. SDK Selection & Deprecation Analysis

### Recommended SDK: `google-genai` (Current & Official)
- **Status**: Current official SDK from Google.
- **Package**: `google-genai` (v2.32.0+ installed in `C:\Users\rohit\.venv\Scripts\python.exe`).
- **Import**: `from google import genai`
- **Supported Models**: `gemini-3.5-flash-lite` (default, highly responsive), `gemini-3.5-flash`, etc.

### Compatibility Issue with Legacy `google-generativeai`
- Google has officially ended support for the legacy `google-generativeai` package, which now emits:
  ```
  FutureWarning: All support for the `google.generativeai` package has ended.
  It will no longer be receiving updates or bug fixes. Please switch to the
  `google.genai` package as soon as possible.
  ```
- **Pipeline Implementation**: This repository defaults to the modern `google-genai` SDK, while retaining a graceful fallback adapter to `google-generativeai` if running in environments where only the legacy package is installed.

---

## 2. Workspace & Environment Configuration

### Windows Environment
- **Workspace**: `c:\Users\rohit\task q1`
- **Python Executable**: `C:\Users\rohit\.venv\Scripts\python.exe`

### Setting API Keys
API keys are loaded securely from environment variables (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) and are never hardcoded.

In Windows PowerShell:
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
```

In Windows Command Prompt (CMD):
```cmd
set GEMINI_API_KEY=your-gemini-api-key-here
```

---

## 3. Web Application Interface (Localhost)

A state-of-the-art interactive web dashboard is included to run extractions visually with live Pydantic validation, dual JSON/Visual card views, preset loaders, and error diagnostics.

### Start the Local Web App:
```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" server.py
```
Then open: **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 4. Deploying to Vercel

The application is structured for instant serverless deployment on Vercel:
- **Serverless API Routes**: Python handlers in [`api/extract.py`](file:///c:/Users/rohit/task%20q1/api/extract.py) and [`api/sample.py`](file:///c:/Users/rohit/task%20q1/api/sample.py).
- **Static Assets**: Frontend served from [`public/index.html`](file:///c:/Users/rohit/task%20q1/public/index.html).
- **Configuration**: Managed via [`vercel.json`](file:///c:/Users/rohit/task%20q1/vercel.json).

### Option A: Deploy via Vercel CLI (Command Line)
1. Run Vercel CLI in this directory:
   ```powershell
   npx vercel
   ```
2. Follow the interactive prompts to link to your Vercel account.
3. Add the Gemini API Key to your Vercel environment:
   ```powershell
   npx vercel env add GEMINI_API_KEY
   ```
   *(Paste your API key when prompted, and select Production, Preview, and Development)*
4. Deploy to production:
   ```powershell
   npx vercel --prod
   ```

### Option B: Deploy via GitHub / Vercel Dashboard
1. Push this repository to GitHub (ensure `.env` remains ignored by `.gitignore`).
2. Go to [vercel.com/new](https://vercel.com/new) and import the repository.
3. In **Environment Variables**, add:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: `your_gemini_api_key_here`
4. Click **Deploy**. Vercel will install `requirements.txt` and launch both the web frontend and Python serverless endpoints.

---

## 5. Project Structure

```text
c:\Users\rohit\task q1\
├── api\
│   ├── extract.py            # Vercel serverless function (/api/extract)
│   └── sample.py             # Vercel serverless function (/api/sample)
├── public\
│   └── index.html            # Static web dashboard for Vercel
├── vercel.json               # Vercel routing configuration
├── extractor.py              # Main extraction pipeline, Pydantic schemas, & CLI
├── server.py                 # Local HTTP web application server (Port 8000)
├── index.html                # Local development copy of web dashboard
├── test_extractor.py         # Unit & integration test suite (12 tests)
├── demo_invalid_rejection.py # Verification of strict rejection of corrupt/invalid data
├── requirements.txt          # Dependencies installed by Vercel & local venv
├── .env                      # Local environment configuration (GEMINI_API_KEY)
├── .gitignore                # Protects secrets and .vercel from git
├── README.md                 # Setup, local run, and deployment documentation
└── samples\
    ├── invoice_sample.txt    # Realistic invoice text input
    └── resume_sample.txt     # Realistic resume / CV text input
```

---

## 5. Key Architectural Features

1. **Strict Pydantic Models**:
   - `ConfigDict(extra="forbid")` prevents unmapped or unexpected data injection.
   - Field constraints: `quantity > 0`, `unit_price >= 0`, `total >= 0`, `min_length=1`.
   - Email format validation with regex matching.
2. **Parameterized Few-Shot Prompt Templates**:
   - Includes schema definitions and realistic input/output demonstrations to steer LLM output.
3. **Resilient JSON Parser**:
   - Automatically cleans markdown fences (` ```json ... ``` `) and conversational preambles/postambles.
4. **Comprehensive Error Handling**:
   - Custom exceptions: `MalformedJSONError`, `SchemaValidationError`, `MissingAPIKeyError`, `LLMAPIError`.
   - Detailed field-level error messages showing location, error type, and invalid value.
5. **Deterministic Offline / Mock Mode**:
   - Supports `--mock` flag to run validation and schema tests offline without burning API quota.

---

## 5. Execution Commands

### A. Run Invoice Extraction

**Live execution (requires `GEMINI_API_KEY`):**
```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" extractor.py --type invoice --file samples/invoice_sample.txt
```

**Offline / Mock execution (deterministic verification):**
```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" extractor.py --type invoice --file samples/invoice_sample.txt --mock
```

---

### B. Run Resume Extraction

**Live execution (requires `GEMINI_API_KEY`):**
```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" extractor.py --type resume --file samples/resume_sample.txt
```

**Offline / Mock execution (deterministic verification):**
```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" extractor.py --type resume --file samples/resume_sample.txt --mock
```

---

### C. Run the Unit Test Suite

Runs all 12 unit tests validating schema enforcement, regex, error handling, and JSON parsing:
```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" -m unittest test_extractor.py -v
```

---

### D. Run the Invalid Data Rejection Demo

Demonstrates strict rejection of 5 critical corruption scenarios:
1. Negative line item quantity.
2. Missing required fields.
3. Malformed email address.
4. Forbidden extra injected fields.
5. Truncated / corrupt JSON.

```powershell
& "C:\Users\rohit\.venv\Scripts\python.exe" demo_invalid_rejection.py
```
