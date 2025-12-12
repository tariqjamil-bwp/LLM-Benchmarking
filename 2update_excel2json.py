# ================================================================================
# HEADER
# ================================================================================
"""
File: 2update_excel2json.py
Author: Dr. Shradha's Research Team
Date: 2025-12-11
Version: 1.1.0
Introduction: Syncs manually edited Excel file to JSON with nested structure,
              preserving verification results and model responses.
"""

# ================================================================================
# IMPORTS
# ================================================================================
import json
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ================================================================================
# FILE DEFINITIONS
# ================================================================================
INPUT_EXCEL_FILE = "data/qna_responses_final.xlsx"  # Input: Manually edited Excel
OUTPUT_JSON_FILE = "data/qna_responses_final.json"  # Output: Nested JSON structure

# ================================================================================
# MAIN PROCESSING
# ================================================================================
def update_json_from_excel():
    """Sync Excel file to JSON with nested structure (responses + correct keys)"""

    # Load Excel file
    df = pd.read_excel(INPUT_EXCEL_FILE)

    # Identify model response columns (exclude metadata and verification columns)
    exclude_columns = ['id', 'level', 'category', 'problem_text', 'problem_latex', 'answer_latex', 'instruction']
    model_columns = []
    for col in df.columns:
        if col not in exclude_columns and not col.endswith('_correct'):
            model_columns.append(col)

    print(f"Model columns found: {model_columns} ({len(model_columns)} total)")

    # Validate and fix _correct columns
    print(f"\n{'='*80}")
    print("VALIDATING VERIFICATION COLUMNS")
    print(f"{'='*80}")

    missing_correct_cols = []
    orphaned_correct_cols = []
    excel_modified = False

    # Ensure each model has a corresponding _correct column
    for model in model_columns:
        correct_col = f"{model}_correct"
        if correct_col not in df.columns:
            print(f"⚠ Missing: {correct_col} (adding empty column)")
            df[correct_col] = pd.Series(dtype='object')
            missing_correct_cols.append(correct_col)
            excel_modified = True
        else:
            print(f"✓ Found: {correct_col}")

    # Check for orphaned _correct columns
    for col in df.columns:
        if col.endswith('_correct'):
            model_name = col[:-8]
            if model_name not in model_columns:
                print(f"⚠ Orphaned: {col} (model '{model_name}' not found)")
                orphaned_correct_cols.append(col)

    # Save Excel if columns were added
    if excel_modified:
        print(f"\n✓ Adding {len(missing_correct_cols)} missing _correct columns to Excel...")
        df.to_excel(INPUT_EXCEL_FILE, index=False)
        print(f"✓ Excel updated: {INPUT_EXCEL_FILE}")

    if orphaned_correct_cols:
        print(f"\n⚠ WARNING: {len(orphaned_correct_cols)} orphaned _correct columns found")

    print(f"{'='*80}\n")

    # Load existing JSON to preserve verification results
    try:
        with open(OUTPUT_JSON_FILE, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    # Build updated JSON with nested structure
    updated_json_data = {}
    metadata_columns = ['level', 'category', 'problem_text', 'problem_latex', 'answer_latex', 'instruction']

    for _, row in df.iterrows():
        question_id = str(row['id'])

        # Initialize with nested structure: {responses: {}, correct: {}}
        updated_json_data[question_id] = {
            "responses": {},
            "correct": {}
        }

        # Add metadata fields at root level
        for meta_col in metadata_columns:
            if meta_col in df.columns and pd.notna(row[meta_col]):
                updated_json_data[question_id][meta_col] = str(row[meta_col])

        # Preserve existing verification results from JSON
        if question_id in existing_data:
            if "correct" in existing_data[question_id] and isinstance(existing_data[question_id]["correct"], dict):
                # New structure: copy from "correct" key
                for model_name, correct_value in existing_data[question_id]["correct"].items():
                    updated_json_data[question_id]["correct"][model_name] = correct_value
            elif "responses" in existing_data[question_id] and isinstance(existing_data[question_id]["responses"], dict):
                # Old nested structure: migrate from responses/{model}_correct
                for key, value in existing_data[question_id]["responses"].items():
                    if key.endswith('_correct'):
                        model_name = key[:-8]
                        updated_json_data[question_id]["correct"][model_name] = value
            else:
                # Flat structure: migrate from root-level {model}_correct
                for key, value in existing_data[question_id].items():
                    if key.endswith('_correct'):
                        model_name = key[:-8]
                        updated_json_data[question_id]["correct"][model_name] = value

        # Add model responses under "responses" key
        for model_col in model_columns:
            if pd.notna(row[model_col]) and str(row[model_col]).lower() != 'nan':
                updated_json_data[question_id]["responses"][model_col] = row[model_col]
            else:
                updated_json_data[question_id]["responses"][model_col] = None

            # Add verification value under "correct" key
            correct_col = f"{model_col}_correct"
            if correct_col in df.columns and pd.notna(row[correct_col]):
                correct_value = row[correct_col]
                # Convert string to boolean if needed
                if isinstance(correct_value, str):
                    correct_value = correct_value.lower() in ['true', '1', 'yes']
                elif isinstance(correct_value, (int, float)):
                    correct_value = bool(correct_value)
                updated_json_data[question_id]["correct"][model_col] = correct_value
            else:
                updated_json_data[question_id]["correct"][model_col] = None

    # Save updated JSON
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(updated_json_data, f, indent=2)

    # Print summary
    print("="*80)
    print("SYNC SUMMARY")
    print("="*80)
    print(f"Questions: {len(updated_json_data)}")

    # Count model responses
    model_counts = {}
    for model in model_columns:
        count = sum(1 for q in updated_json_data.values()
                   if q.get("responses", {}).get(model) is not None)
        model_counts[model] = count

    print("\nModel responses:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}/{len(updated_json_data)}")

    # Count verifications
    verification_counts = {}
    for q_data in updated_json_data.values():
        for model_name, value in q_data.get("correct", {}).items():
            if model_name not in verification_counts:
                verification_counts[model_name] = 0
            if value is not None:
                verification_counts[model_name] += 1

    if verification_counts:
        print("\nVerifications preserved:")
        for model, count in sorted(verification_counts.items()):
            print(f"  {model}: {count}/{len(updated_json_data)}")

    print(f"\nOutput: {OUTPUT_JSON_FILE}")
    print("="*80)

    return updated_json_data

# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main():
    """Main execution function"""
    try:
        update_json_from_excel()
        print("\nProcess completed successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_EXCEL_FILE}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
