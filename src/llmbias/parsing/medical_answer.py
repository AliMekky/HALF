"""Historical notebook cells [297]; structural extraction, unchanged scientific logic."""
import pandas as pd
import re

def extract_final_choice(text):
    text = str(text).strip()
    patterns = ['The answer is\\s*([A-E])\\b', 'Correct answer[:\\s]*([A-E])\\b', 'Answer[:\\s]*([A-E])\\b', 'Closest answer is\\s*([A-E])\\b']
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if re.fullmatch('[A-E]', last_line):
            return last_line
    first_line = lines[0] if lines else ''
    if re.fullmatch('[A-E]', first_line):
        return first_line
    return None

def process_file(input_path, output_csv_path):
    df = pd.read_csv(input_path)
    df['clean_answer'] = df['answer'].apply(extract_final_choice)
    df.to_csv(output_csv_path, index=False)
