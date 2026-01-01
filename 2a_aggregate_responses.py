# ================================================================================
# HEADER
# ================================================================================
"""
File: 1a_aggregate_responses.py
Author: Dr. Shradha's Research Team
Date: 2026-01-01
Version: 1.0.0
Introduction: Aggregates model responses from multiple Excel files.
              Merges qna_responses.xlsx with all qna_responses_*.xlsx files
              (excluding _final variants) into qna_responses_final.xlsx
"""

# ================================================================================
# IMPORTS
# ================================================================================
import pandas as pd
import os
import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ================================================================================
# FILE DEFINITIONS
# ================================================================================
INPUT_FILE = "data/qna_responses.xlsx"           # Code 1 output
OUTPUT_FILE = "data/qna_responses_final.xlsx"    # Aggregated output
PATTERN = "data/qna_responses_*.xlsx"            # Model response files

# ================================================================================
# AGGREGATION FUNCTION
# ================================================================================
def aggregate_model_responses():
    """Aggregate all model response files into single Excel file"""

    print("="*80)
    print("MODEL RESPONSE AGGREGATION")
    print("="*80)

    # Load main file from Code 1
    print(f"\n1. Loading main file: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"✗ Error: {INPUT_FILE} not found")
        return False

    main_df = pd.read_excel(INPUT_FILE)
    print(f"   ✓ Loaded: {len(main_df)} rows, {len(main_df.columns)} columns")

    # Find all model response files
    print(f"\n2. Discovering model response files...")
    all_files = sorted(glob.glob(PATTERN))

    # Exclude main input and output files
    exclude_names = {
        'qna_responses.xlsx',
        'qna_responses_final.xlsx',
        'qna_responses_merged.xlsx'
    }
    model_files = [f for f in all_files if os.path.basename(f) not in exclude_names]

    if not model_files:
        print(f"   ⚠ No additional model files found")
        print(f"   ℹ Skipping merge, will use main file as-is")
    else:
        print(f"   ✓ Found {len(model_files)} model file(s):")

        # Merge each model file
        for model_file in model_files:
            filename = os.path.basename(model_file)
            print(f"\n3. Merging: {filename}")

            try:
                model_df = pd.read_excel(model_file)
                print(f"   ✓ Loaded: {len(model_df)} rows, {len(model_df.columns)} columns")

                # Get model columns (exclude metadata)
                metadata_cols = {'id', 'instruction'}
                model_cols = [col for col in model_df.columns if col not in metadata_cols]

                if not model_cols:
                    print(f"   ⚠ No model columns found (only id/instruction)")
                    continue

                print(f"   ✓ Model columns: {model_cols}")

                # Merge by 'id'
                for model_col in model_cols:
                    if model_col in main_df.columns:
                        print(f"   ⚠ Column '{model_col}' already exists (skipping)")
                        continue

                    # Merge by id
                    main_df = main_df.merge(
                        model_df[['id', model_col]],
                        on='id',
                        how='left'
                    )

                    # Add corresponding _correct column
                    correct_col = f"{model_col}_correct"
                    main_df[correct_col] = pd.Series(dtype='object')

                    print(f"   ✓ Added: {model_col}")
                    print(f"   ✓ Added: {correct_col}")

            except Exception as e:
                print(f"   ✗ Error processing {filename}: {str(e)}")
                continue

    # Save aggregated file
    print(f"\n4. Saving aggregated file...")
    main_df.to_excel(OUTPUT_FILE, index=False)
    print(f"   ✓ Saved: {OUTPUT_FILE}")

    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATION SUMMARY")
    print(f"{'='*80}")
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"\nFinal structure:")
    print(f"  - Rows: {len(main_df)}")
    print(f"  - Columns: {len(main_df.columns)}")

    # List model columns
    exclude_columns = {'id', 'subcategory', 'problem_text', 'problem_latex', 'answer_latex', 'instruction'}
    model_columns = [col for col in main_df.columns if col not in exclude_columns and not col.endswith('_correct')]

    if model_columns:
        print(f"\nModel columns ({len(model_columns)} total):")
        for i, col in enumerate(sorted(model_columns), 1):
            print(f"  {i}. {col}")

    print(f"\n✓ Aggregation completed successfully!")
    print(f"{'='*80}")

    return True

# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main():
    """Main execution function"""
    try:
        aggregate_model_responses()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
