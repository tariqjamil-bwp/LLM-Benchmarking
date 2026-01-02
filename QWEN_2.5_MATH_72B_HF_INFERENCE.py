#!/usr/bin/env python3
"""
QWEN 2.5-MATH 72B HF INFERENCE - HuggingFace Interface Integration
Collects responses from Qwen2.5-Math via HuggingFace Spaces demo interface
Auto-detects input field using multiple strategies with robust fallbacks
"""

import pandas as pd
import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ================================================================================
# LOGGING
# ================================================================================
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# Configuration
DEMO_URL = "https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo"
INPUT_FILE = "data/ds2_qna.xlsx"
OUTPUT_FILE = "data/qna_responses_qwen2.5_math_72b.xlsx"
CHECKPOINTS_DIR = "checkpoints"
MODEL_COLUMN_NAME = "Qwen 2.5-Math 72B"

# Ensure checkpoints directory exists
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ================================================================================
# SELENIUM SETUP
# ================================================================================
def setup_firefox_driver():
    """Setup Firefox WebDriver with geckodriver"""
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Uncomment for headless:
    # options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    return driver

# ================================================================================
# ELEMENT DETECTION STRATEGIES
# ================================================================================
def find_input_field_strategy_1(driver, wait):
    """Strategy 1: Look for textareas"""
    try:
        textareas = driver.find_elements(By.CSS_SELECTOR, "textarea")
        if len(textareas) > 0:
            return textareas[0], "textarea"
    except:
        pass
    return None, None

def find_input_field_strategy_2(driver, wait):
    """Strategy 2: Look for input[type=text]"""
    try:
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        if len(inputs) > 0:
            return inputs[0], "input[type=text]"
    except:
        pass
    return None, None

def find_input_field_strategy_3(driver, wait):
    """Strategy 3: Look for elements with role='textbox'"""
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, "[role='textbox']")
        if len(elements) > 0:
            return elements[0], "[role='textbox']"
    except:
        pass
    return None, None

def find_input_field_strategy_4(driver, wait):
    """Strategy 4: Look for contenteditable divs"""
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, "[contenteditable='true']")
        if len(elements) > 0:
            return elements[0], "[contenteditable='true']"
    except:
        pass
    return None, None

def find_input_field_strategy_5(driver, wait):
    """Strategy 5: Wait for any input-like element to be visible"""
    try:
        # Wait for ANY element to become visible
        input_field = wait.until(
            EC.visibility_of_any_elements_located((By.TAG_NAME, "input")),
            timeout=10
        )
        if input_field:
            return input_field[0], "visible input[waited]"
    except:
        pass
    return None, None

def find_input_field_strategy_6(driver, wait):
    """Strategy 6: Search in iframes if they exist"""
    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframes:
            try:
                driver.switch_to.frame(iframe)
                textareas = driver.find_elements(By.TAG_NAME, "textarea")
                if len(textareas) > 0:
                    return textareas[0], "textarea[in iframe]"
                inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
                if len(inputs) > 0:
                    return inputs[0], "input[in iframe]"
                driver.switch_to.default_content()
            except:
                driver.switch_to.default_content()
                continue
    except:
        pass
    return None, None

def find_input_field(driver, wait, logger):
    """Try multiple strategies to find input field"""
    strategies = [
        find_input_field_strategy_1,
        find_input_field_strategy_2,
        find_input_field_strategy_3,
        find_input_field_strategy_4,
        find_input_field_strategy_5,
        find_input_field_strategy_6,
    ]

    for i, strategy in enumerate(strategies):
        try:
            elem, desc = strategy(driver, wait)
            if elem:
                logger.info(f"✓ Found input field using strategy {i+1}: {desc}")
                return elem
        except Exception as e:
            logger.debug(f"Strategy {i+1} failed: {str(e)}")

    return None

# ================================================================================
# RESPONSE CLEANING FUNCTION
# ================================================================================
def clean_response_text(raw_response):
    """Remove UI text from beginning and ending - PRESERVE ALL INTERNAL FORMATTING"""

    cleaned = raw_response

    # STRATEGY 1: Remove the EXACT repeating UI header block (appears consistently at start)
    ui_header_block = "📖 Qwen2.5-Math Demo\nThis WebUI is based on Qwen2-VL for OCR and Qwen2.5-Math for mathematical reasoning. You can input either images or texts of mathematical or arithmetic problems.\nUpload\nUpload\nSketch\nUpload\nDrop Image Here\n- or -\nClick to Upload\ninput your question\nClear\nSubmit\n"

    if cleaned.startswith(ui_header_block):
        cleaned = cleaned[len(ui_header_block):]
    else:
        # Try to find it anywhere in first 500 chars (in case formatting varies)
        idx = cleaned.find(ui_header_block)
        if idx >= 0 and idx < 500:
            cleaned = cleaned[:idx] + cleaned[idx + len(ui_header_block):]

    # STRATEGY 2: Fallback - if Strategy 1 didn't work, find FIRST math keyword
    # (handles cases where UI header format differs)
    if len(cleaned) > 1500 and cleaned.startswith("📖"):  # Still has garbage
        math_keywords = ["To find", "To solve", "Let ", "First", "We ", "Substituting", "Given", "The matrix", "The determinant", "For the", "To compute"]
        earliest_idx = -1
        for keyword in math_keywords:
            idx = cleaned.find(keyword)
            if idx >= 0:
                if earliest_idx == -1 or idx < earliest_idx:
                    earliest_idx = idx
        if earliest_idx > 0:
            cleaned = cleaned[earliest_idx:]

    # Remove footer blocks from ending
    footer_patterns = [
        "\nUse via API\n·\nBuilt with Gradio\n·\nSettings",
        "\n·\nBuilt with Gradio\n·\nSettings",
        "\nUse via API",
        "Use via API",
    ]

    for pattern in footer_patterns:
        if pattern in cleaned:
            idx = cleaned.rfind(pattern)
            if idx >= 0:
                cleaned = cleaned[:idx]
                break

    # Only remove trailing whitespace LINES, not internal spaces/newlines
    lines = cleaned.split('\n')
    while lines and not lines[-1].strip():
        lines.pop()

    cleaned = '\n'.join(lines)

    # DISABLED: Colon formatting was causing response truncation
    # # Format mathematical output: add newlines around colons for better readability
    # # If a line contains text and colon (like "is: value"), split on colon
    # formatted_lines = []
    # for line in cleaned.split('\n'):
    #     if ':' in line and not line.rstrip().endswith(':'):
    #         # Line has text AFTER the colon - split it
    #         colon_idx = line.find(':')
    #         before_colon = line[:colon_idx + 1]  # Include the colon
    #         after_colon = line[colon_idx + 1:].lstrip()  # Text after colon, trim spaces
    #
    #         formatted_lines.append(before_colon)
    #         if after_colon:  # Only add if there's text after
    #             formatted_lines.append(after_colon)
    #     else:
    #         formatted_lines.append(line)
    #
    # cleaned = '\n'.join(formatted_lines)

    return cleaned

# ================================================================================
# MAIN COLLECTION FUNCTION
# ================================================================================
def collect_responses_via_selenium(qids_to_process=None):
    """
    Automate Qwen2.5-Math-Demo to collect responses (V2 - Improved)

    Args:
        qids_to_process (list): List of qids (ids) to process.
                                If None or empty list, process all qids from input file.
    """
    logger = setup_logging()

    # Load input file
    logger.info(f"Loading data from {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    logger.info(f"Loaded {len(df)} questions")

    # Load latest checkpoint and create response cache dictionary
    cached_responses = {}
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.xlsx')]) if os.path.exists(CHECKPOINTS_DIR) else []

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, latest_checkpoint)
        logger.info(f"Found checkpoint: {latest_checkpoint} - Reading cached responses")
        df_checkpoint = pd.read_excel(checkpoint_path)

        # Build dictionary of all responses from checkpoint
        for idx, row in df_checkpoint.iterrows():
            qid = row['id']
            response = row[MODEL_COLUMN_NAME]
            # Only cache valid responses (not errors, not empty, not None)
            if pd.notna(response) and response != "" and not str(response).startswith("Error"):
                cached_responses[qid] = response

        logger.info(f"Loaded {len(cached_responses)} cached responses from checkpoint")
    else:
        logger.info("No checkpoint found - starting fresh")

    # Filter to specified qids or use all
    if qids_to_process and len(qids_to_process) > 0:
        df_to_process = df[df['id'].isin(qids_to_process)].copy()
        df_to_process = df_to_process.set_index('id').loc[qids_to_process].reset_index()
        logger.info(f"Processing {len(df_to_process)} specified qids")
    else:
        df_to_process = df.copy()
        logger.info(f"Processing all {len(df_to_process)} qids from input file")

    # Create output file structure
    df_output = df[['id', 'instruction']].copy()
    df_output[MODEL_COLUMN_NAME] = None

    # Pre-populate output with cached responses
    for qid, response in cached_responses.items():
        matching_rows = df_output[df_output['id'] == qid]
        if len(matching_rows) > 0:
            output_idx = matching_rows.index[0]
            df_output.at[output_idx, MODEL_COLUMN_NAME] = response

    logger.info(f"Output structure ready: {len(df_output)} questions, {len(cached_responses)} pre-populated")

    print("\n" + "=" * 80)
    print("QWEN 2.5-MATH SELENIUM SCRAPER V2 (IMPROVED)")
    print("=" * 80)
    print(f"URL: {DEMO_URL}")
    print(f"Total questions: {len(df_output)}")
    print(f"Cached from checkpoint: {len(cached_responses)}")
    print(f"Need to scrape: {len(df_to_process)}")
    print("=" * 80 + "\n")

    driver = None
    try:
        driver = setup_firefox_driver()
        logger.info("Initializing Firefox WebDriver...")

        # Open page
        logger.info(f"Opening {DEMO_URL}")
        driver.get(DEMO_URL)

        # Initial quick wait, then dynamically check if page is ready
        logger.info("Initial wait (10 seconds)...")
        time.sleep(10)

        # Create wait object
        wait = WebDriverWait(driver, 20)

        # Wait dynamically for input field to be accessible
        logger.info("Checking if page is ready (input field accessible)...")
        field_found = False
        for attempt in range(30):  # Try for up to 30 seconds
            try:
                # Check main document
                textareas = driver.find_elements(By.TAG_NAME, "textarea")
                if len(textareas) > 0:
                    field_found = True
                    logger.info("✓ Input field found in main document")
                    break

                # Check iframes
                iframes = driver.find_elements(By.TAG_NAME, "iframe")
                for iframe in iframes:
                    try:
                        driver.switch_to.frame(iframe)
                        textareas = driver.find_elements(By.TAG_NAME, "textarea")
                        driver.switch_to.default_content()
                        if len(textareas) > 0:
                            field_found = True
                            logger.info("✓ Input field found in iframe")
                            break
                    except:
                        driver.switch_to.default_content()

                if field_found:
                    break
                time.sleep(1)
            except:
                time.sleep(1)

        if not field_found:
            logger.warning("Input field not found after waiting - proceeding anyway")

        # Process each question
        for process_idx, (_, row) in enumerate(df_to_process.iterrows(), 1):
            qid = row['id']
            instruction = row['instruction']

            # Find output row index
            output_idx = df_output[df_output['id'] == qid].index[0]

            print(f"\n{process_idx}/{len(df_to_process)}: {qid}")
            logger.info(f"Processing {qid} ({process_idx}/{len(df_to_process)})")

            # Check if response already in cache
            if qid in cached_responses:
                print(f"  ↻ Using cached response ({len(cached_responses[qid])} chars)")
                logger.info(f"Using cached response from checkpoint: {len(cached_responses[qid])} chars")
                df_output.at[output_idx, MODEL_COLUMN_NAME] = cached_responses[qid]
                time.sleep(0.5)  # Small delay
                continue

            try:
                # Find input field using multiple strategies
                logger.info("Finding input field...")
                input_field = find_input_field(driver, wait, logger)

                if not input_field:
                    raise Exception("Could not find input field with any strategy")

                # Clear field
                driver.execute_script("arguments[0].value = '';", input_field)
                time.sleep(0.5)

                # Type instruction
                input_field.clear()
                input_field.send_keys(instruction)
                logger.info(f"Entered instruction: {instruction[:50]}...")
                time.sleep(1)

                # Find and click submit button
                logger.info("Finding submit button...")
                submit_btn = None

                # Try multiple button detection strategies
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    btn_text = btn.text.lower()
                    if any(x in btn_text for x in ["submit", "run", "send", "execute", "generate"]):
                        submit_btn = btn
                        break

                if not submit_btn:
                    # Try to find clickable button near the input
                    if len(buttons) > 0:
                        submit_btn = buttons[-1]  # Usually last button

                if not submit_btn:
                    raise Exception("Could not find submit button")

                # Scroll button into view and click
                driver.execute_script("arguments[0].scrollIntoView(true);", submit_btn)
                time.sleep(0.5)
                submit_btn.click()
                logger.info("Clicked Submit button")

                # Wait for response - detect when model STOPS responding (response stabilizes)
                # Strategy: Poll response length, when it stops growing for 3 checks = done. Then wait 2 more seconds.
                logger.info("Waiting for response generation to complete (monitoring for stability)...")
                response_text = "No response extracted"
                wait_start = time.time()
                max_wait = 150  # Increased from 90 to 150 seconds for slower questions
                check_interval = 1  # Check every 1 second
                stable_threshold = 5  # Increased from 3 to 5 checks (5 seconds stable = done)
                stable_count = 0
                last_response_length = 0

                # Known UI text to skip - filter out upload/button areas
                ui_text_to_skip = [
                    "This WebUI is based on Qwen2-VL for OCR and Qwen2.5-Math for mathematical reasoning",
                    "You can input either images or texts of mathematical or arithmetic problems",
                    "Upload",
                    "Sketch",
                    "Drop Image Here",
                    "Click to Upload",
                    "input your question",
                    "Clear",
                    "Submit",
                ]

                try:
                    response_found = False

                    while not response_found and (time.time() - wait_start) < max_wait:
                        try:
                            # Look for the longest substantial text element
                            all_elements = driver.find_elements(By.TAG_NAME, "*")

                            best_response = None
                            best_length = 0

                            for elem in all_elements:
                                try:
                                    text = elem.text.strip()

                                    # Skip if too short, contains question
                                    if len(text) < 100:  # Lower threshold to catch responses early
                                        continue
                                    if instruction[:30] in text:
                                        continue

                                    # Check if this is PURE UI text (only UI keywords, no real content)
                                    # If text is made up entirely of UI keywords repeated, skip it
                                    ui_keyword_count = sum(text.count(ui) for ui in ui_text_to_skip)
                                    ui_keyword_ratio = ui_keyword_count / max(len(text.split()), 1)  # Ratio of UI words to total words

                                    # If more than 50% of words are UI keywords, it's UI text
                                    if ui_keyword_ratio > 0.5:
                                        logger.debug(f"Skipping UI-heavy text (ratio={ui_keyword_ratio:.2f}): {text[:50]}...")
                                        continue

                                    # This is a valid response - keep the longest one
                                    if len(text) > best_length:
                                        best_response = text
                                        best_length = len(text)
                                except:
                                    continue

                            # Check if response length is stable (not growing)
                            if best_response:
                                if best_length == last_response_length:
                                    # Length hasn't changed, count towards stability
                                    stable_count += 1
                                    logger.debug(f"Response stable: {best_length} chars, stability_count={stable_count}/{stable_threshold}")
                                else:
                                    # Length changed, reset counter
                                    stable_count = 0
                                    last_response_length = best_length
                                    logger.debug(f"Response growing: {best_length} chars")

                                # If stable for enough checks, model has finished
                                if stable_count >= stable_threshold:
                                    logger.info(f"Response stabilized at {best_length} chars, waiting 2 more seconds to confirm completion...")
                                    time.sleep(2)  # Wait 2 more seconds to ensure it's complete

                                    # Check if response is actually complete (ends properly)
                                    final_best_response = None
                                    final_best_length = 0
                                    for elem in driver.find_elements(By.TAG_NAME, "*"):
                                        try:
                                            text = elem.text.strip()
                                            if len(text) > 100 and instruction[:30] not in text:
                                                ui_keyword_count = sum(text.count(ui) for ui in ui_text_to_skip)
                                                ui_keyword_ratio = ui_keyword_count / max(len(text.split()), 1)
                                                if ui_keyword_ratio <= 0.5 and len(text) > final_best_length:
                                                    final_best_response = text
                                                    final_best_length = len(text)
                                        except:
                                            pass

                                    if final_best_response:
                                        response_text = final_best_response
                                    else:
                                        response_text = best_response

                                    response_found = True
                                    elapsed = time.time() - wait_start
                                    logger.info(f"✓ Response captured after {elapsed:.1f}s: {len(response_text)} chars")

                                    # Log last 100 chars to see how response ends
                                    last_100 = response_text[-100:].replace('\n', ' ')
                                    logger.debug(f"Response ends with: ...{last_100}")

                            # Wait before next check
                            if not response_found:
                                time.sleep(check_interval)

                        except Exception as e:
                            logger.debug(f"Check attempt failed: {str(e)}")
                            time.sleep(check_interval)

                    if not response_found:
                        logger.warning(f"No response found after {max_wait} seconds (stability check)")
                        response_text = "No response extracted - timeout"

                except Exception as e:
                    logger.warning(f"Error during response extraction: {str(e)}")
                    response_text = "Response extraction failed - check manually"

                # Clean up the response text before saving
                if response_text and response_text != "No response extracted":
                    cleaned_response = clean_response_text(response_text)
                    original_len = len(response_text)
                    cleaned_len = len(cleaned_response)
                    logger.info(f"Response cleaned: {original_len} -> {cleaned_len} chars (removed {original_len - cleaned_len} UI chars)")
                    response_text = cleaned_response

                df_output.at[output_idx, MODEL_COLUMN_NAME] = response_text
                cached_responses[qid] = response_text  # Add to cache for future resumption
                print(f"  ✓ Got response ({len(response_text)} chars)")
                logger.info(f"Response received (cleaned): {len(response_text)} chars")

                # Save checkpoint every 3 questions
                if process_idx % 3 == 0:
                    total_completed = len([v for v in df_output[MODEL_COLUMN_NAME] if pd.notna(v) and v != "" and not str(v).startswith("Error")])
                    checkpoint_file = f"{CHECKPOINTS_DIR}/qwen_checkpoint_{total_completed}.xlsx"
                    df_output.to_excel(checkpoint_file, index=False)
                    logger.info(f"Checkpoint saved: {checkpoint_file} ({total_completed} total completed)")

            except Exception as e:
                error_msg = f"Error processing {qid}: {str(e)}"
                print(f"  ✗ {error_msg}")
                logger.error(error_msg)
                df_output.at[output_idx, MODEL_COLUMN_NAME] = f"Error: {str(e)}"

            # Rate limiting
            time.sleep(2)

        # Save final output
        logger.info(f"Saving output to {OUTPUT_FILE}")
        df_output.to_excel(OUTPUT_FILE, index=False)
        print(f"\n✓ Saved: {OUTPUT_FILE}")
        logger.info(f"Saved final output: {OUTPUT_FILE}")

        # Print summary
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)
        successful = df_output[MODEL_COLUMN_NAME].notna().sum()
        print(f"Total questions: {len(df_output)}")
        print(f"Successful responses: {successful}")
        print(f"Failed/Missing: {len(df_output) - successful}")
        print("=" * 80)

        return df_output

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Error: {str(e)}")
        return None

    finally:
        if driver:
            logger.info("Closing WebDriver...")
            driver.quit()

# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main(qids_to_process=None):
    """
    Main execution function

    Args:
        qids_to_process (list): List of qids to process. If None or empty, process all qids.
    """
    try:
        logger = setup_logging()
        logger.info("Starting Qwen2.5-Math Selenium scraper V2 (Improved)")

        df_output = collect_responses_via_selenium(qids_to_process=qids_to_process)

        if df_output is not None:
            print("\n✓ Process completed successfully!")
            logger.info("Process completed successfully")
        else:
            print("\n✗ Process failed")
            logger.error("Process failed")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        logger.info("Process interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    print("""
================================================================================
QWEN 2.5-MATH SELENIUM SCRAPER V2 (IMPROVED)
================================================================================

IMPROVEMENTS:
1. Multiple element detection strategies (6 different approaches)
2. Extended wait times for JavaScript rendering
3. Better error messages showing which strategy worked
4. Iframe support
5. Improved button finding logic
6. Better response extraction
7. Checkpoint saving every 3 questions

USAGE:
  python3 QWEN_2.5_MATH_SELENIUM_SCRAPER_V2.py

OUTPUT:
  File: data/qna_responses_qwen2.5_math_72b.xlsx
  Columns: id, instruction, Qwen 2.5-Math 72B

NOTES:
- Browser window will open - watch it to debug if needed
- First run will take ~5-10 min per question due to extended waits
- Check terminal output for which strategy worked
================================================================================
""")

    main()  # Process all qids from ds2_qna.xlsx
