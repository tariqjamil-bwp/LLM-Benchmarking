# ================================================================================
# HEADER
# ================================================================================
"""
File: models.py
Author: Dr. Shradha's Research Team
Date: 2025-12-11
Version: 1.0.0
Introduction: Provides API communication functions for accessing LLMs via
              OpenRouter API and model configuration for verification agent.
"""

# ================================================================================
# IMPORTS
# ================================================================================
import os
import requests
from google.adk.models.lite_llm import LiteLlm

# ================================================================================
# CONFIGURATION
# ================================================================================
TEMPERATURE = 0.0

# Model configurations
MODEL_CONFIGS = [
    {'api_id': 'qwen/qwen-2.5-72b-instruct', 'column_name': 'Qwen_72B'},
    {'api_id': 'deepseek/deepseek-v3.2', 'column_name': 'DeepSeek-V3'},
    {'api_id': 'mistralai/mistral-large-2512', 'column_name': 'Mistral Large'},
    {'api_id': 'meta-llama/llama-3.3-70b-instruct', 'column_name': 'Llama 3.3 70B'},
    {'api_id': 'openai/gpt-4o', 'column_name': 'GPT-4o'}
]
# ================================================================================
# API FUNCTIONS
# ================================================================================
def get_model_response(prompt, model_id, api_key, temperature=0.0, max_tokens=2000):
    """Get response from model via OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: No choices in response - {result}"
    else:
        return f"Error: HTTP {response.status_code} - {response.text}"


def get_model_for_agent(modelnum=1):
    """
    Get the LLM model instance for verification agent.

    Args:
        modelnum: 1 for OpenAI (default), 2 for DeepSeek (fallback)

    Returns:
        LiteLlm model instance configured with appropriate API key
    """
    if modelnum == 1:
        model = LiteLlm(model="openai/gpt-5-mini", api_key=os.environ.get("OPENAI_API_KEYx"))
    elif modelnum == 2:
        model = LiteLlm(model="deepseek/deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEYx"))
    else:
        raise ValueError(f"Invalid modelnum: {modelnum}. Use 1 for OpenAI or 2 for DeepSeek.")

    return model
