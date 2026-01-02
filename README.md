# LLM Response Collection & Verification Pipeline

A production-ready pipeline for collecting responses from multiple Large Language Models, verifying their correctness, and generating analysis reports. Supports API-based collection, web scraping, local inference, and manual entry.

## 📌 What's New in v1.1.0

**Enhanced API Response Tracking:** Captures OpenRouter metadata (finish_reason, token counts) for definitive truncation detection and optimization analysis.

**Restructured JSON Format:** Separate "metadata" key prevents metadata fields from leaking into response detection, eliminating architectural bugs.

**Improved Verification:** Simplified verification system with dynamic metadata extraction works seamlessly with new JSON structure.

See [Latest Enhancements](#latest-enhancements-v110) below for details.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Collection Methods

Choose one or combine multiple collection methods:

#### 1. API Collection (Multiple Models via OpenRouter)
```bash
export OPENROUTER_API_KEY="your_api_key_here"
python3 1model_response_collector.py
```
Collects from: GPT-4o, DeepSeek, Mistral, Llama, Qwen. Resume-enabled via checkpoints.

#### 2. HuggingFace Interface (Qwen2.5-Math Demo)
```bash
# Install geckodriver
brew install geckodriver              # macOS
# OR: sudo apt-get install firefox-geckodriver  # Linux

python3 QWEN_2.5_MATH_72B_HF_INFERENCE.py
```
Collects from Qwen2.5-Math HuggingFace demo interface with intelligent resume capability.
(requires firefox gecko driver installed)

#### 3. Colab Inference (Qwen-7B)
```bash
jupyter notebook QWEN_2.5_MATH_7B_HF.ipynb
```
Runs Qwen-7B on Google Colab. No API keys required.

#### 4. Manual Entry
Edit `data/qna_responses_final.xlsx` to add responses for additional models directly.

### Complete Pipeline

After collecting responses:

```bash
# Optional: Aggregate responses from multiple sources
python3 2a_aggregate_responses.py      # Merge Code 1 + Selenium + Manual + Other APIs
                                       # Creates qna_responses_final.xlsx

# Required: Convert to JSON and verify
python3 2update_excel2json.py          # Sync Excel to structured JSON
python3 3model_response_verifier.py    # Verify correctness with agent
```

**Note on 2a_aggregate_responses.py:**
- **Optional** - Only use if you have responses from multiple collection sources
- **When to use:**
  - Combining Code 1 (API) responses with Selenium scraper responses
  - Merging manual entries with API responses
  - Adding responses from alternative/custom APIs
- **Skipped if:** Only using Code 1 (API collection) alone

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Questions Dataset (Excel)                               │
│  Columns: id, instruction, problem_text, answer_latex, ...      │
└────────────────────┬────────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
   ╔════════════╗ ╔════════════╗ ╔════════════╗ ╔════════════╗
   │ API        │ │ Web        │ │ Local      │ │ Manual     │
   │ Collection │ │ Scraping   │ │ Inference  │ │ Entry      │
   ╚════════════╝ ╚════════════╝ ╚════════════╝ ╚════════════╝
      │              │              │              │
      ▼              ▼              ▼              ▼
┌────────────────────────────────────────────────────────────────┐
│  STAGE 1: Multiple Collection Methods (Choose One or Combine)  │
├────────────────────────────────────────────────────────────────┤
│  API Collection (1model_response_collector.py)                 │
│      • OpenRouter: GPT-4o, DeepSeek, Mistral, Llama, Qwen      │
│      • Resume via JSON checkpoint                              │
│      • Output: qna_responses.xlsx                              │
├────────────────────────────────────────────────────────────────┤
│  HuggingFace Interface (QWEN_2.5_MATH_72B_HF_INFERENCE.py)     │
│      • Collects from Qwen2.5-Math HF demo interface            │
│      • Dynamic page load detection + iframe handling           │
│      • Stability-based response detection                      │
│      • Smart cache resume from checkpoints                     │
│      • Output: qna_responses_qwen2.5_math_72b.xlsx             │
├────────────────────────────────────────────────────────────────┤
│  Local/Colab Inference (QWEN_2.5_MATH_7B_HF.ipynb)             │
│      • Qwen-7B local/Colab inference via Transformers          │
│      • No API needed - runs on your hardware                   │
│      • Output: qna_responses_qwen_7b_hf.xlsx                   │
├────────────────────────────────────────────────────────────────┤
│  Manual Entry (qna_responses_final.xlsx)                       │
│      • Add more models by editing Excel manually               │
│      • Paste responses from any source                         │
│      • Aggregation (next stage) will merge to JSON              │
└────────────────────┬───────────────────────────────────────────┘
                     │ (Multiple response Excel files)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1a: Response Aggregation [OPTIONAL]                      │
│  • 2a_aggregate_responses.py                                    │
│  • Merges responses from multiple collection sources            │
│  • Combines: Code 1 + Selenium + Manual + Other APIs           │
│  • Only needed if using multiple collection methods            │
│  • Outputs: qna_responses_final.xlsx (merged)                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Data Sync (Excel → JSON)                              │
│  • 2update_excel2json.py                                        │
│  • Syncs Excel responses to JSON with nested structure          │
│  • Column validation & metadata key creation                    │
│  • Outputs: qna_responses_final.json                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Verification (LLM-as-Judge)                           │
│  • 3model_response_verifier.py                                  │
│  • Uses LLM agent to verify correctness of all models           │
│  • Async processing with resume capability                      │
│  • Outputs: qna_verified_responses.json + Excel with scores     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Verified Dataset                                        │
│  • qna_verified_responses.json (with correctness flags)         │
│  • qna_verified_responses_final.xlsx (human-readable)           │
│  • Analysis-ready for all models                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Latest Enhancements (v1.1.0)

### 1. API Response Metadata Tracking (Code 1 & models.py)

**Problem Solved:** Previous approach used heuristics to detect incomplete responses (unreliable).

**Solution:** Now captures OpenRouter API metadata for definitive truncation detection.

**What's Captured:**
- `finish_reason`:
  - `"stop"` = response completed normally
  - `"length"` = response hit max_tokens limit
  - `"error"` = API error occurred
- `completion_tokens`: Tokens used for response generation
- `total_tokens`: Total tokens including prompt

**Improvement:** Increased `max_tokens` from 4000 to 10000 for more complete responses.

**Example response with metadata:**
```json
{
  "content": "To find the determinant...",
  "finish_reason": "stop",
  "completion_tokens": 1410,
  "total_tokens": 1534
}
```

**Impact:** Can now definitively identify which responses were truncated and optimize token usage across models.

---

### 2. JSON Structure with Separate Metadata Key (Code 2)

**Problem Solved:** Metadata fields ('instruction', 'subcategory') were leaking into model responses dict, causing false model detection.

**Solution:** Separate "metadata" key in JSON structure with dynamic field detection from Excel headers.

**Old Structure (Mixed/Problematic):**
```json
{
  "question_id": {
    "responses": {"model": "response"},
    "problem_text": "...",        ← Metadata at root
    "instruction": "..."          ← Can leak into responses!
  }
}
```

**New Structure (Clean):**
```json
{
  "question_id": {
    "metadata": {
      "problem_text": "...",
      "problem_latex": "...",
      "answer_latex": "...",
      "instruction": "...",
      "subcategory": "...",
      "level": "3",
      "category": "computational"
    },
    "responses": {
      "Qwen-72B": "response...",
      "DeepSeek-V3": "response..."
    },
    "correct": {
      "Qwen-72B": true,
      "DeepSeek-V3": false
    }
  }
}
```

**Key Changes:**
- Changed `BASE_QNA_FILE` from ds4_qna.json to ds4_qna.xlsx (source of truth)
- Dynamic metadata detection from Excel column headers (no hardcoding)
- Backwards compatible migration from old flat structure
- Prevents bugs where metadata leaks into model response detection

**Impact:** Architectural fix - metadata can never be mistaken for a model name.

---

### 3. Simplified Verification Logic (Code 3)

**Changes:**
- Updated to read metadata from separate "metadata" key
- Changed `answer_latex` access to `question["metadata"]["answer_latex"]`
- Updated Excel export to read metadata columns from metadata key
- Simplified response existence checks (removed unnecessary complexity)
- Dynamic metadata column extraction from JSON structure

**Impact:** Verification pipeline now aligns with clean JSON architecture, reducing bugs and maintenance burden.

---

## Project Structure

```
project/
├── Collection Scripts
│   ├── 1model_response_collector.py        # API-based collection (OpenRouter)
│   ├── QWEN_2.5_MATH_72B_HF_INFERENCE.py   # HuggingFace interface (Selenium)
│   ├── QWEN_2.5_MATH_7B_HF.ipynb           # Local inference (Jupyter)
│   │
│   ├── 2a_aggregate_responses.py           # [OPTIONAL] Aggregate multi-source responses
│   └── 2update_excel2json.py               # Merge all sources and sync to JSON
│
├── Verification
│   └── 3model_response_verifier.py         # Verify response correctness
│
├── Shared
│   ├── models.py                           # Configuration & API utilities
│   ├── requirements.txt                    # Dependencies
│   └── monitor_checkpoints.py              # Progress monitoring
│
├── data/
│   ├── ds2_qna.xlsx                       # Input questions
│   ├── qna_responses.xlsx                 # API collection output
│   ├── qna_responses_qwen2.5_math_72b.xlsx # Scraper output
│   ├── qna_responses_qwen_7b_hf.xlsx      # Local inference output
│   ├── qna_responses_final.xlsx           # Merged responses (editable)
│   ├── qna_responses_final.json           # Final JSON (Stage 2 output)
│   └── qna_verified_responses.json        # Verified responses (Stage 3 output)
│
├── checkpoints/
│   ├── qwen_checkpoint_3.xlsx
│   ├── qwen_checkpoint_6.xlsx
│   └── ... (auto-saved every 3 questions)
│
└── README.md
```

## HuggingFace Interface Details

The HF Inference script (`QWEN_2.5_MATH_72B_HF_INFERENCE.py`) automatically collects responses from the Qwen2.5-Math HuggingFace demo interface.

### How It Works

**Dynamic Page Loading:** Waits for the interface to load completely, detecting when input fields are accessible in iframes before processing.

**Intelligent Response Detection:** Monitors response generation in real-time, detecting when the model finishes outputting (stability timeout: 5 seconds, max wait: 150 seconds per question).

**Smart Resume:** Loads the latest checkpoint into memory. When restarting, it instantly skips previously completed questions and continues from where it left off.

**Response Cleaning:** Automatically removes UI elements while preserving mathematical formatting, LaTeX, and newlines.

**Progress Checkpoints:** Saves a checkpoint every 3 questions for recovery in case of interruption.

### Running the Collection

```bash
python3 QWEN_2.5_MATH_72B_HF_INFERENCE.py
```

The browser window opens automatically. You can watch progress in real-time. If interrupted, run the command again to resume from the latest checkpoint.

### Configuration

Edit at the top of the script:

```python
DEMO_URL = "https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo"
INPUT_FILE = "data/ds2_qna.xlsx"
OUTPUT_FILE = "data/qna_responses_qwen2.5_math_72b.xlsx"
max_wait = 150                  # Seconds to wait per question
stable_threshold = 5            # Stability checks (1 second each)
```

---

## 🚀 Usage & Customization

### For Your Own Benchmark

#### Step 1: Prepare Your Dataset

Create an Excel file (`data/your_benchmark.xlsx`) with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique problem identifier |
| `instruction` | string | **Full prompt to send to models** |
| `problem_text` | string | Problem description (optional) |
| `problem_latex` | string | LaTeX format (optional) |
| `answer_latex` | string | **Ground truth answer (used for verification)** |

**Minimal example:**
```excel
id               | instruction                                        | answer_latex
Q1_easy          | What is 2+2?                                       | 4
Q2_medium        | Solve: x² - 5x + 6 = 0                             | x = 2, x = 3
Q3_hard          | Integrate: ∫x sin(x) dx                            | -x cos(x) + sin(x) + C
```

#### Step 2: Configure Models

Edit `models.py` to add/remove models:

```python
MODEL_CONFIGS = [
    {'api_id': 'openai/gpt-4o', 'column_name': 'GPT-4o'},
    {'api_id': 'deepseek/deepseek-v3.2', 'column_name': 'DeepSeek-V3'},
    {'api_id': 'your-custom-model', 'column_name': 'YourModel'},  # Add custom models
]
```

**Available providers** (via OpenRouter):
- OpenAI: `openai/gpt-4o`, `openai/gpt-4-turbo`
- DeepSeek: `deepseek/deepseek-v3.2`, `deepseek/deepseek-chat`
- Anthropic: `anthropic/claude-3-opus`
- Meta: `meta-llama/llama-3.3-70b-instruct`
- Mistral: `mistralai/mistral-large-2512`
- Qwen: `qwen/qwen-2.5-72b-instruct`

[Full model list](https://openrouter.ai/docs/models)

#### Step 3: Run the Pipeline

```bash
# Update file paths in each script to match your benchmark
# Modify INPUT_FILE, OUTPUT_FILE, JSON_FILE paths

# Run stages in order
python3 1model_response_collector.py     # Collect responses
python3 2update_excel2json.py            # Sync to JSON
python3 3model_response_verifier.py      # Verify with agent
```

#### Step 4: Generate Analysis

After pipeline completes, create analysis scripts using the verified JSON:

```python
import json
import pandas as pd

# Load verified results
with open('data/qna_verified_responses.json', 'r') as f:
    results = json.load(f)

# Calculate accuracy per model
accuracies = {}
for problem_id, problem_data in results.items():
    for model_name, is_correct in problem_data['correct'].items():
        if model_name not in accuracies:
            accuracies[model_name] = {'correct': 0, 'total': 0}
        if is_correct is not None:
            accuracies[model_name]['total'] += 1
            if is_correct:
                accuracies[model_name]['correct'] += 1

# Display results
for model, counts in accuracies.items():
    pct = (counts['correct'] / counts['total'] * 100) if counts['total'] > 0 else 0
    print(f"{model}: {counts['correct']}/{counts['total']} ({pct:.1f}%)")
```

## 🔌 API Integration Approaches

### Unified API Gateway: OpenRouter

Use OpenRouter for unified access to multiple models with a single API key:

```python
# In models.py
def get_model_response(prompt, model_id, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={"model": model_id, "messages": [{"role": "user", "content": prompt}]}
    )
    return response.json()['choices'][0]['message']['content']
```

**Pros:** Single API key, 200+ models, unified interface
**Cons:** Third-party dependency, potential rate limits

---

### Direct API Integration

Integrate each model's official API directly:

#### OpenAI (GPT-4o)
```python
import openai

def get_openai_response(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000
    )
    return response.choices[0].message.content
```

#### DeepSeek
```python
import requests

def get_deepseek_response(prompt, api_key):
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )
    return response.json()['choices'][0]['message']['content']
```

#### Anthropic (Claude)
```python
from anthropic import Anthropic

def get_claude_response(prompt, api_key):
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-opus-20250219",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

#### Meta (Llama via Together/Replicate)
```python
# Via Together AI
import requests

def get_llama_response(prompt, api_key):
    response = requests.post(
        "https://api.together.xyz/inference",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "prompt": prompt,
            "max_tokens": 2000
        }
    )
    return response.json()['output']['choices'][0]['text']
```

#### Google (Gemini)
```python
import google.generativeai as genai

def get_gemini_response(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text
```

#### Mistral
```python
from mistralai.client import MistralClient

def get_mistral_response(prompt, api_key):
    client = MistralClient(api_key=api_key)
    response = client.chat(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### Qwen (Alibaba)
```python
import requests

def get_qwen_response(prompt, api_key):
    response = requests.post(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen-plus",
            "input": {"messages": [{"role": "user", "content": prompt}]}
        }
    )
    return response.json()['output']['text']
```

**Pros:** Direct control, no intermediaries, better rate limits, cost savings
**Cons:** Need separate API keys per model, manage multiple endpoints

---

### LiteLLM Abstraction (Current for Verification)

Use LiteLLM for unified verification agent interface:

```python
from litellm import completion

def verify_response(model_response, correct_answer, model="gpt-4"):
    response = completion(
        model=model,
        messages=[{
            "role": "user",
            "content": f"Compare: {model_response} vs {correct_answer}"
        }]
    )
    return response.choices[0].message.content
```

**Pros:** Unified interface, easy fallback support, logging
**Cons:** Another dependency, slight latency overhead

---

## 🔧 Advanced Features

### Resume Capability

All stages support resuming from checkpoints:

**Stage 1 (Collector):**
- Loads `qna_responses.json` if exists
- Skips already-processed questions
- Updates after each response

**Stage 3 (Verifier):**
- Loads `qna_verified_responses.json` if exists
- Skips already-verified models
- Updates after each verification

### Manual Verification Override

Edit the intermediate Excel file to manually verify responses, then run Stage 2 to sync:

```python
# In qna_responses_final.xlsx
# Set columns: [Model]_correct = 1 (correct) or 0 (incorrect)
# Then run:
python3 2update_excel2json.py
```

### Custom Verification Agent

Replace the verification logic in `3model_response_verifier.py`:

```python
# Modify create_verification_agent() function
def create_verification_agent(model):
    return LlmAgent(
        model=model,
        instruction="""
        Your custom verification instruction here.
        Compare model_response against correct_answer.
        Return JSON: {"is_correct": bool, "explanation": str}
        """
    )
```

---

## ✅ Verification Approaches

The current Stage 3 uses Google ADK's LlmAgent for verification, but multiple approaches are available based on your requirements:

### Google ADK LlmAgent (Current - Recommended for Complex Verification)

Use the built-in LlmAgent framework:

```python
from google.adk.models.lite_llm import LlmAgent

def create_verification_agent(model="gpt-4o"):
    """Create LLM agent for verification"""
    return LlmAgent(
        model=model,
        instruction="""
        Compare the model response against the correct answer.
        Consider: mathematical correctness, logical reasoning, formatting.
        Return JSON: {"is_correct": bool, "confidence": float, "explanation": str}
        """
    )

# Usage in Stage 3
agent = create_verification_agent()
result = agent.run(f"Response: {model_response}\nCorrect: {correct_answer}")
```

**Pros:** Framework handles async/sessions, structured output, error handling
**Cons:** Requires google-adk package, external dependency
**Best for:** Production pipelines, complex verification logic, multiple agents
**Cost:** ~$0.0001 per verification call (depending on model)

---

### Simple Direct LLM Comparison (Lightweight)

Skip the agent framework and use direct API calls:

```python
import requests
import json

def verify_response_simple(response, correct_answer, api_key, model="gpt-4o"):
    """Simple verification without framework"""
    headers = {"Authorization": f"Bearer {api_key}"}

    prompt = f"""Compare these two answers:
    Model Response: {response}
    Correct Answer: {correct_answer}

    Reply with JSON only: {{"is_correct": true/false, "reason": "..."}}"""

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()
    text = result['choices'][0]['message']['content']

    # Parse JSON from response
    import re
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"is_correct": False, "reason": "Failed to parse"}
```

**Pros:** No dependencies, simple to understand, fast
**Cons:** Manual error handling, no async support, basic output parsing
**Best for:** Quick prototyping, simple correctness checks
**Cost:** ~$0.0001 per verification call


## 📊 Data Formats

### Input Format (Excel)

```
id | instruction | problem_text | answer_latex | ...
```

### Intermediate Format (JSON - Stage 2 Output)

```json
{
  "Q1_easy": {
    "id": "Q1_easy",
    "instruction": "What is 2+2?",
    "answer_latex": "4",
    "responses": {
      "GPT-4o": "The answer is 4",
      "DeepSeek-V3": "4"
    },
    "correct": {
      "GPT-4o": null,
      "DeepSeek-V3": null
    }
  }
}
```

### Final Format (JSON - Stage 3 Output)

```json
{
  "Q1_easy": {
    "id": "Q1_easy",
    "instruction": "What is 2+2?",
    "answer_latex": "4",
    "responses": {
      "GPT-4o": "The answer is 4",
      "DeepSeek-V3": "4"
    },
    "correct": {
      "GPT-4o": true,
      "DeepSeek-V3": true
    },
    "explanations": {
      "GPT-4o": "Response correctly identifies answer as 4",
      "DeepSeek-V3": "Response correctly identifies answer as 4"
    }
  }
}
```

### Excel Output Format

| id | instruction | answer_latex | GPT-4o | GPT-4o_correct | DeepSeek-V3 | DeepSeek-V3_correct |
|----|-------------|--------------|--------|----------------|-------------|---------------------|
| Q1_easy | What is 2+2? | 4 | The answer is 4 | 1 | 4 | 1 |

## 🔑 Configuration Guide

### Environment Variables (OS Level)

Set environment variables in your terminal/shell:

```bash
# Linux/Mac - Add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Windows - Use PowerShell or Command Prompt
setx OPENROUTER_API_KEY "sk-or-..."
setx OPENAI_API_KEY "sk-..."
```

Then reload your terminal and run scripts:
```bash
python3 1model_response_collector.py
```

**Pros:** Secure, standard practice, works globally
**Cons:** Not portable between machines, requires terminal restart

---

### .env File (Recommended for Development)

Create a `.env` file in your project root:

```env
# .env - API Keys & Credentials
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
QWEN_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Optional: File paths (if you want to configure via .env)
INPUT_FILE=data/your_questions.xlsx
OUTPUT_FILE=data/your_responses.xlsx
DATA_DIR=data/

# Optional: Model configuration
TEMPERATURE=0.0
MAX_TOKENS=2000
VERIFICATION_MODEL=gpt-4o
```

Then modify your scripts to load from `.env`:

```python
# At the top of each script
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now access variables
api_key = os.getenv("OPENROUTER_API_KEY")
input_file = os.getenv("INPUT_FILE", "data/default_questions.xlsx")
```

**Installation:**
```bash
pip install python-dotenv
```

**Pros:** Portable, safe (add .env to .gitignore), easy to switch configs
**Cons:** Requires python-dotenv package, .env file must be in project root

**Important:** Add `.env` to `.gitignore` to prevent committing API keys:

```bash
# .gitignore
.env
.env.local
.env.*.local
*.pyc
__pycache__/
data/
```

---

### File Path Configuration

Edit these variables in each script:

**1model_response_collector.py:**
```python
INPUT_FILE = "data/your_questions.xlsx"
OUTPUT_FILE = "data/your_responses.xlsx"
JSON_FILE = "data/your_responses_checkpoint.json"
```

**2update_excel2json.py:**
```python
INPUT_EXCEL_FILE = "data/qna_responses_final.xlsx"
OUTPUT_JSON_FILE = "data/qna_responses_final.json"
```

**3model_response_verifier.py:**
```python
RESPONSES_JSON_FILE = "data/qna_responses_final.json"
VERIFIED_JSON_FILE = "data/qna_verified_responses.json"
OUTPUT_EXCEL_FILE = "data/qna_verified_responses_final.xlsx"
```

## 📋 Requirements

```
pandas>=1.3.0
openpyxl>=3.6.0
requests>=2.26.0
torch>=2.0.0
transformers>=4.30.0
google-adk>=0.1.0  # For verification agent (optional)
google-cloud-aiplatform>=1.25.0  # For Gemini API (optional)
```

Install with:
```bash
pip install -r requirements.txt
```

## 🎯 Real-World Examples

### Example 1: Math Problem Evaluation

```python
# Evaluate math models on standard tests
# Input: 100 calculus problems with LaTeX answers
# Models: GPT-4o, Claude-3, DeepSeek-V3
# Output: Accuracy breakdown by topic

python3 1model_response_collector.py
python3 2update_excel2json.py
python3 3model_response_verifier.py

# Then analyze by topic in Analysis/
```

### Example 2: Code Generation

```python
# Evaluate code generation models
# Input: 50 coding challenges with expected outputs
# Models: Claude-3-Opus, Mistral-Large, GPT-4o

# Modify instruction to include code testing:
# "Solve this problem and provide working Python code"
# Modify verification to check if code runs without error

python3 1model_response_collector.py
python3 2update_excel2json.py
# Custom verification for code execution
python3 3model_response_verifier.py
```

### Example 3: Knowledge Assessment

```python
# Evaluate domain knowledge models
# Input: 200 domain-specific questions with correct answers
# Models: Domain-trained vs general models

# Run standard pipeline and compare accuracy
python3 1model_response_collector.py
python3 2update_excel2json.py
python3 3model_response_verifier.py

# Analyze results by difficulty level and domain
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENROUTER_API_KEY not found` | Set environment variable: `export OPENROUTER_API_KEY="..."` |
| `FileNotFoundError: data/questions.xlsx` | Update `INPUT_FILE` path in scripts to match your data location |
| `API Error: HTTP 429` | Rate limiting. Increase delays in Stage 1 (line ~134) |
| `Verification fails to parse JSON` | Check verification agent instruction returns valid JSON |
| `Out of memory on GPU` | Reduce batch size or use CPU in Notebook Stage 4 |
| `Empty responses.json` | Check API key and model availability on OpenRouter |

## 📈 Performance Optimization


## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙋 Support & Contributing

### Questions?
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review example configurations in this README
3. Check function docstrings in source files

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request with description of changes

### Extending the Pipeline
- Add custom verification logic in Stage 3
- Support additional API providers in `models.py`
- Implement parallel processing for faster execution
- Add automatic analysis generation

## 🎓 Citations

If you use this pipeline in research, cite as:

```bibtex
@software{llm_pipeline_2025,
  title={LLM Response Collection & Verification Pipeline},
  author={Dr. Shradha Research Team},
  year={2025},
  url={https://github.com/tariqjamil-bwp/LLM-Benchmarking}
}
```

---

**Last Updated**: December 31, 2025
**Version**: 2.0.0
**Status**: Production-ready with Selenium Scraping & Multi-Source Collection
