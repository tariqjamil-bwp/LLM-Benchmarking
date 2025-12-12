# LLM Response Collection & Verification Pipeline

A modular, production-ready pipeline for collecting responses from multiple Large Language Models (LLMs), verifying their correctness, and generating analysis reports. Designed for extensibility - easily adapt to your own evaluation benchmarks.

## 🎯 Quick Start

```bash
# 1. Setup environment
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_api_key_here"

# 2. Run the pipeline
python3 1model_response_collector.py
python3 2update_excel2json.py
python3 3model_response_verifier.py
```

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Questions Dataset (Excel)                               │
│  Columns: id, instruction, problem_text, answer_latex, ...      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Response Collection (OpenRouter API)                  │
│  • 1model_response_collector.py                                 │
│  • Collects responses from N models                             │
│  • Resume capability via JSON checkpoints                       │
│  • Outputs: qna_responses.xlsx + responses.json                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Data Sync (Excel → JSON)                              │
│  • 2update_excel2json.py                                        │
│  • Syncs manually edited Excel to JSON                          │
│  • Nested structure: {responses, correct, explanations}         │
│  • Column validation & auto-creation                            │
│  • Outputs: qna_responses_final.json                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Verification (LLM-as-Judge)                           │
│  • 3model_response_verifier.py                                  │
│  • Uses LLM agent to verify correctness                         │
│  • Async processing with resume capability                      │
│  • Outputs: qna_verified_responses.json + Excel with scores     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Verified Dataset                                        │
│  • qna_verified_responses.json (with correctness flags)         │
│  • qna_verified_responses_final.xlsx (human-readable)           │
│  • Analysis-ready data                                          │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
project/
├── 1model_response_collector.py      # Stage 1: Collect responses
├── 2update_excel2json.py             # Stage 2: Sync Excel→JSON
├── 3model_response_verifier.py       # Stage 3: Verify responses
├── 4_qwen_inference_pipeline.ipynb   # Optional: Qwen-7B local inference
├── models.py                          # Shared config & API functions
├── requirements.txt                   # Python dependencies
├── data/
│   ├── questions.xlsx                # Input: Your questions
│   ├── qna_responses.xlsx            # Stage 1 output
│   ├── qna_responses_final.json      # Stage 2 output
│   └── qna_verified_responses.json   # Stage 3 output
├── Analysis/                          # Generated charts & summaries
└── README.md                          # This file
```

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
id               | instruction                                          | answer_latex
Q1_easy          | What is 2+2?                                        | 4
Q2_medium        | Solve: x² - 5x + 6 = 0                             | x = 2, x = 3
Q3_hard          | Integrate: ∫x sin(x) dx                             | -x cos(x) + sin(x) + C
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

## 🔌 API Integration Options

### Option 1: OpenRouter (Current - Convenience)

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

### Option 2: Direct API Integration (Recommended for Production)

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

### Option 3: LiteLLM Abstraction (Current for Verification)

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

### Option 4: Custom Abstraction Layer (Recommended)

Build your own unified interface:

```python
# api_clients.py
class APIClient:
    """Base class for model APIs"""
    def get_response(self, prompt, **kwargs):
        raise NotImplementedError

class OpenAIClient(APIClient):
    def __init__(self, api_key):
        self.api_key = api_key
    def get_response(self, prompt, **kwargs):
        # OpenAI implementation
        pass

class DeepSeekClient(APIClient):
    def __init__(self, api_key):
        self.api_key = api_key
    def get_response(self, prompt, **kwargs):
        # DeepSeek implementation
        pass

class ClaudeClient(APIClient):
    def __init__(self, api_key):
        self.api_key = api_key
    def get_response(self, prompt, **kwargs):
        # Claude implementation
        pass

# Usage
clients = {
    'gpt-4o': OpenAIClient(os.environ['OPENAI_API_KEY']),
    'deepseek-v3': DeepSeekClient(os.environ['DEEPSEEK_API_KEY']),
    'claude-3-opus': ClaudeClient(os.environ['ANTHROPIC_API_KEY'])
}

response = clients['gpt-4o'].get_response("Your prompt here")
```

**Pros:** Complete control, extensible, no external dependencies
**Cons:** More boilerplate code, need to maintain each API integration

---

### Recommended Approach by Use Case

| Use Case | Recommended | Reason |
|----------|------------|--------|
| **Quick Prototyping** | OpenRouter | Single API key, 200+ models |
| **Production Evaluation** | Direct APIs | Cost savings, reliability, no rate limits |
| **Multi-Model Agent** | Custom Layer | Flexibility, easy to switch/fallback |
| **Research Comparison** | LiteLLM | Unified logging, experiment tracking |
| **Educational** | Direct APIs | Learn model-specific quirks |

---

## 🔧 Advanced Features

### Resume Capability

All stages support resuming from checkpoints:

**Stage 1 (Collector):**
- Loads `responses.json` if exists
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

## ✅ Verification Agent Alternatives

The current Stage 3 uses Google ADK's LlmAgent for verification, but you can choose different approaches based on your needs:

### Option A: Google ADK LlmAgent (Current - Recommended for Complex Verification)

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

### Option B: Simple Direct LLM Comparison (Lightweight)

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

---

### Option C: Pydantic + Direct API (Type-Safe)

Use Pydantic for validation and structured outputs:

```python
from pydantic import BaseModel
from typing import Optional
import requests
import json

class VerificationResult(BaseModel):
    is_correct: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    error_type: Optional[str] = None  # e.g., "calculation", "logic", "format"

def verify_with_pydantic(response, correct_answer, api_key, model="gpt-4o"):
    """Verification with type validation"""
    headers = {"Authorization": f"Bearer {api_key}"}

    prompt = f"""Verify this response strictly.
    Model: {response}
    Correct: {correct_answer}

    JSON: {{"is_correct": bool, "confidence": 0.0-1.0, "reason": "...", "error_type": null or string}}"""

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    api_response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = api_response.json()
    text = result['choices'][0]['message']['content']

    # Validate with Pydantic
    try:
        verification = VerificationResult(**json.loads(text))
        return verification
    except Exception as e:
        return VerificationResult(
            is_correct=False,
            confidence=0.0,
            reason=f"Validation failed: {str(e)}"
        )
```

**Pros:** Type-safe, auto-validation, IDE support, cleaner code
**Cons:** Requires Pydantic, more boilerplate, slower parsing
**Best for:** Large-scale deployments, data quality concerns
**Cost:** ~$0.0001 per verification call

---

### Option D: Rule-Based Verification (No LLM)

Skip LLM entirely for deterministic checks:

```python
import re
from difflib import SequenceMatcher

def verify_response_rules(response, correct_answer):
    """Rule-based verification without LLM"""

    # Normalize responses
    resp_norm = response.lower().strip()
    corr_norm = correct_answer.lower().strip()

    # Exact match
    if resp_norm == corr_norm:
        return {"is_correct": True, "method": "exact_match"}

    # Extract numerical answers
    resp_nums = re.findall(r'-?\d+\.?\d*', resp_norm)
    corr_nums = re.findall(r'-?\d+\.?\d*', corr_norm)

    if resp_nums and corr_nums:
        try:
            # Check with ±0.01 tolerance
            if abs(float(resp_nums[0]) - float(corr_nums[0])) < 0.01:
                return {"is_correct": True, "method": "numeric_match"}
        except:
            pass

    # Similarity check
    similarity = SequenceMatcher(None, resp_norm, corr_norm).ratio()
    if similarity > 0.95:
        return {"is_correct": True, "method": "similarity", "score": similarity}

    return {"is_correct": False, "method": "no_match", "similarity": similarity}
```

**Pros:** Instant, no API calls, no costs, deterministic
**Cons:** Limited to simple answers, no semantic understanding
**Best for:** Multiple choice, exact numerical answers, quick screening
**Cost:** Free

---

### Option E: Hybrid Approach (Smart Routing)

Use rules first, LLM for complex cases:

```python
def verify_response_hybrid(response, correct_answer, api_key, threshold=0.5):
    """Use rules first, then LLM if needed"""

    # Step 1: Try rule-based verification
    rule_result = verify_response_rules(response, correct_answer)

    if rule_result['is_correct']:
        return {
            "is_correct": True,
            "method": "rule_based",
            "confidence": 0.99
        }

    # Step 2: If uncertain, use LLM
    if rule_result.get('similarity', 0) > threshold:
        llm_result = verify_with_pydantic(response, correct_answer, api_key)
        return {
            "is_correct": llm_result.is_correct,
            "method": "llm_verification",
            "confidence": llm_result.confidence,
            "reason": llm_result.reason
        }

    # Step 3: Clear mismatch
    return {
        "is_correct": False,
        "method": "rule_based",
        "confidence": 0.99,
        "reason": f"Similarity too low: {rule_result.get('similarity', 0):.2f}"
    }
```

**Pros:** Cost-efficient, fast for obvious cases, LLM for edge cases
**Cons:** More complex logic, threshold tuning needed
**Best for:** Large-scale benchmarks, cost-sensitive deployments
**Cost:** ~$0.00001 per verification (mostly rules, occasional LLM)

---

### Verification Approach Comparison

| Aspect | Option A (LlmAgent) | Option B (Simple) | Option C (Pydantic) | Option D (Rules) | Option E (Hybrid) |
|--------|-------------------|------------------|-------------------|-----------------|------------------|
| **Speed** | ~2-3s per call | ~2-3s per call | ~2-3s per call | <100ms | ~1s average |
| **Accuracy** | Very high (95%+) | High (90%+) | Very high (95%+) | Low (70%) | High (90%+) |
| **Cost/1000** | $0.10 | $0.10 | $0.10 | Free | $0.01 |
| **Setup Time** | 15 min | 5 min | 10 min | 5 min | 20 min |
| **Parallelizable** | Yes (async) | Yes | Yes | Yes | Yes |
| **Error Handling** | Excellent | Good | Excellent | Good | Good |
| **Best Use Case** | Production | Prototyping | Data quality | QA screening | Scale evaluation |

---

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

### Option 1: Environment Variables (OS Level)

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

### Option 2: .env File (Recommended for Development)

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

### Option 3: .env.example (Share Configuration Template)

Create a `.env.example` file for developers:

```env
# .env.example - Copy this to .env and fill in your credentials
OPENROUTER_API_KEY=sk-or-your-key-here
OPENAI_API_KEY=sk-your-key-here
DEEPSEEK_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
MISTRAL_API_KEY=your-key-here
QWEN_API_KEY=your-key-here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json

INPUT_FILE=data/questions.xlsx
OUTPUT_FILE=data/responses.xlsx
VERIFICATION_MODEL=gpt-4o
TEMPERATURE=0.0
```

Share `.env.example` in repo, but not `.env`:

```bash
# In README
cp .env.example .env
# Then edit .env with your actual API keys
```

---

### Option 4: Config File (YAML/JSON)

For more complex configurations, use a config file:

```yaml
# config.yaml
api_keys:
  openrouter: sk-or-...
  openai: sk-...
  deepseek: sk-...

files:
  input: data/questions.xlsx
  output: data/responses.xlsx
  data_dir: data/

models:
  - name: GPT-4o
    api_id: openai/gpt-4o
    provider: openai
  - name: DeepSeek-V3
    api_id: deepseek/deepseek-v3.2
    provider: deepseek

generation:
  temperature: 0.0
  max_tokens: 2000
  timeout: 30
```

Load in Python:

```python
import yaml
import os

config_path = os.getenv("CONFIG_PATH", "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

api_key = config['api_keys']['openrouter']
input_file = config['files']['input']
```

**Pros:** Complex configurations, environment-specific variants
**Cons:** More setup, harder to keep secrets secure

---

### Option 5: Command-Line Arguments (CLI)

Pass configuration via command-line:

```python
# collector_cli.py
import argparse
import os

parser = argparse.ArgumentParser(description='Collect LLM responses')
parser.add_argument('--input', default='data/questions.xlsx', help='Input file')
parser.add_argument('--output', default='data/responses.xlsx', help='Output file')
parser.add_argument('--api-key', required=True, help='OpenRouter API key')
parser.add_argument('--temperature', type=float, default=0.0, help='Model temperature')
parser.add_argument('--models', nargs='+', default=['gpt-4o', 'deepseek-v3'], help='Models to test')

args = parser.parse_args()

# Use args.input, args.output, etc.
```

Run with:

```bash
python3 1model_response_collector.py \
  --input data/my_questions.xlsx \
  --output data/my_responses.xlsx \
  --api-key sk-or-... \
  --models gpt-4o deepseek-v3 mistral-large
```

**Pros:** Flexible, scriptable, good for CI/CD
**Cons:** Long command lines, hard to manage many parameters

---

### Recommended Configuration Strategy

```
Development:     Use .env file (python-dotenv)
CI/CD Pipeline:  Use environment variables
Production:      Use config file + environment variables
Sharing:         Commit .env.example, not .env
```

### Complete Setup Example

1. **Create `.env` file:**
   ```bash
   cp .env.example .env
   nano .env  # Edit with your API keys
   ```

2. **Update script header:**
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   # Load from .env
   API_KEY = os.getenv("OPENROUTER_API_KEY")
   INPUT_FILE = os.getenv("INPUT_FILE", "data/questions.xlsx")
   OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/responses.xlsx")
   ```

3. **Run script:**
   ```bash
   python3 1model_response_collector.py
   ```

4. **Add to .gitignore:**
   ```bash
   echo ".env" >> .gitignore
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

### Parallel Model Processing
Currently sequential. To parallelize (advanced):
```python
# Modify Stage 1 to use ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_model, config) for config in model_configs]
```

### Batch Verification
Modify Stage 3 to verify multiple questions per agent call:
```python
# Group questions and verify in batches
batch_size = 5
for i in range(0, len(questions), batch_size):
    batch = questions[i:i+batch_size]
    await verify_batch(batch)
```

## 🔄 Integration with Other Tools

### Convert to Hugging Face Dataset
```python
from datasets import Dataset

with open('data/qna_verified_responses.json') as f:
    data = json.load(f)

# Convert to HF format
ds = Dataset.from_dict({
    'id': [...],
    'instruction': [...],
    'correct': [...],
    'explanations': [...]
})

ds.push_to_hub('username/my-benchmark')
```

### Export to CSV for Analytics
```python
import json
import pandas as pd

with open('data/qna_verified_responses.json') as f:
    data = json.load(f)

# Flatten and export
records = []
for qid, qdata in data.items():
    for model, is_correct in qdata['correct'].items():
        records.append({
            'question_id': qid,
            'model': model,
            'correct': is_correct,
            'explanation': qdata['explanations'].get(model, '')
        })

pd.DataFrame(records).to_csv('results.csv', index=False)
```

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
  author={Your Team},
  year={2025},
  url={https://github.com/yourusername/llm-eval-pipeline}
}
```

---

**Last Updated**: December 12, 2025
**Version**: 1.1.0
**Status**: Production-ready
