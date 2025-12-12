# ================================================================================
# HEADER
# ================================================================================
"""
File: 1model_response_collector.py
Author: Dr. Shradha's Research Team
Date: 2025-12-11
Version: 1.1.0
Introduction: Collects responses from multiple LLM models via OpenRouter API
              with resume capability and progress tracking.
"""

# ================================================================================
# IMPORTS
# ================================================================================
import pandas as pd
import time
import logging
import os
import json
from models import get_model_response, MODEL_CONFIGS
# ================================================================================
# FILE DEFINITIONS
# ================================================================================
INPUT_FILE = "data/ds2_qna.xlsx"      # Input: Questions dataset
OUTPUT_FILE = "data/qna_responses.xlsx"  # Output: Excel with collected responses
JSON_FILE = "data/responses.json"     # Output: JSON for resume capability

# ================================================================================
# CONFIGURATION
# ================================================================================
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# ================================================================================
# MAIN PROCESSING
# ================================================================================
def process_model_responses(model_configs):
    """Collect model responses with resume capability"""
    logger = setup_logging()

    if not model_configs:
        # default Model configurations
        model_configs = [
            {'api_id': 'qwen/qwen-2.5-72b-instruct', 'column_name': 'Qwen_72B'},
            {'api_id': 'deepseek/deepseek-v3.2', 'column_name': 'DeepSeek-V3'},
            {'api_id': 'mistralai/mistral-large-2512', 'column_name': 'Mistral Large'},
            {'api_id': 'meta-llama/llama-3.3-70b-instruct', 'column_name': 'Llama 3.3 70B'},
            {'api_id': 'openai/gpt-4o', 'column_name': 'GPT-4o'}
        ]

    # Validate API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not found")
        return

    # Load input data
    logger.info(f"Loading data from {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    logger.info(f"Loaded {len(df)} questions")

    # Load or initialize responses JSON for resume capability
    if os.path.exists(JSON_FILE):
        logger.info(f"Loading existing responses from {JSON_FILE}")
        with open(JSON_FILE, 'r') as f:
            responses = json.load(f)
        logger.info("Existing responses loaded - resume mode active")
    else:
        responses = {}
        logger.info("Starting fresh - no existing responses found")

    # Initialize response structure for all questions and models
    for index, row in df.iterrows():
        question_id = str(row['id'])
        if question_id not in responses:
            responses[question_id] = {}
        for config in model_configs:
            if config['column_name'] not in responses[question_id]:
                responses[question_id][config['column_name']] = None

    # Process each model sequentially
    for config in model_configs:
        model_api_id = config['api_id']
        model_column_name = config['column_name']

        logger.info(f"Starting collection for model: {model_column_name}")

        processed_count = 0
        for index, row in df.iterrows():
            question_id = str(row['id'])

            # Skip if response already exists
            if responses[question_id][model_column_name] is not None:
                processed_count += 1
                logger.info(f"[{processed_count}/{len(df)}] Skipping {question_id} (already processed)")
                continue

            processed_count += 1
            logger.info(f"[{processed_count}/{len(df)}] Processing {question_id}")

            instruction = row['instruction']

            try:
                response = get_model_response(instruction, model_api_id, api_key, max_tokens=1000)

                if response and len(response.strip()) > 0:
                    if "Error:" in response or "HTTP" in response:
                        logger.warning(f"API returned error: {response[:100]}...")
                    else:
                        logger.info(f"Got response ({len(response)} chars)")
                    responses[question_id][model_column_name] = response
                else:
                    logger.warning("Got empty response, keeping as None")

            except Exception as e:
                logger.error(f"Error: {str(e)}")
                responses[question_id][model_column_name] = f"ERROR: {str(e)}"

            # Save progress after each response
            with open(JSON_FILE, 'w') as f:
                json.dump(responses, f)

            # Progress checkpoint every 10 questions
            if processed_count % 10 == 0:
                logger.info(f"Progress: {processed_count}/{len(df)} questions")

        logger.info(f"Completed {model_column_name}")
        time.sleep(2)  # Rate limiting between models

    # Generate final Excel output
    logger.info("Generating Excel output...")
    result_df = df.copy()

    # Add model response columns and empty verification columns
    for config in model_configs:
        model_column_name = config['column_name']

        # Add response column if missing
        if model_column_name not in result_df.columns:
            result_df[model_column_name] = pd.Series(dtype='object')

        # Populate responses from JSON
        for index, row in result_df.iterrows():
            question_id = str(row['id'])
            if question_id in responses and model_column_name in responses[question_id]:
                existing_value = row.get(model_column_name, "")
                if pd.isna(existing_value) or existing_value == "" or str(existing_value) == "nan":
                    response_value = responses[question_id][model_column_name]
                    if response_value is not None:
                        result_df.at[index, model_column_name] = response_value

        # Add empty verification column for manual/automated verification
        correct_column_name = f"{model_column_name}_correct"
        if correct_column_name not in result_df.columns:
            result_df[correct_column_name] = pd.Series(dtype='object')

    result_df.to_excel(OUTPUT_FILE, index=False)
    logger.info(f"Excel output saved: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    for config in model_configs:
        model_column_name = config['column_name']
        if model_column_name in result_df.columns:
            filled_count = result_df[model_column_name].notna().sum()
            print(f"  {model_column_name}: {filled_count}/{len(result_df)} responses")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("="*80)

    return OUTPUT_FILE

# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main():
    """Main execution function"""
    try:
        process_model_responses(model_configs=MODEL_CONFIGS)
        print("\nProcess completed successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
