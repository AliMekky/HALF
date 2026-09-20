"""Historical notebook cells [302]; structural extraction, unchanged scientific logic."""
import json
import csv
from pathlib import Path

def parse_custom_id(custom_id: str):
    """
    Extract (index, gender, ethnicity) from: request-<index>-<...>-<gender>-<ethnicity>
    """
    parts = custom_id.split('-')
    if len(parts) < 3:
        return (None, None, None)
    try:
        idx = int(parts[1])
    except ValueError:
        idx = parts[1]
    gender = parts[-2] if len(parts) >= 2 else None
    ethnicity = parts[-1] if len(parts) >= 1 else None
    return (idx, gender, ethnicity)

def extract_decision(obj: dict):
    """
    Navigate: response → body → choices[0] → message → content
    Normalize as 'Admit' or 'Reject'.
    """
    try:
        txt = obj['response']['body']['choices'][0]['message']['content']
        return txt.strip().title()
    except (KeyError, IndexError, TypeError):
        return None

def process_file(input_path, output_csv_path):
    INPUT_FILE = Path(input_path)
    OUTPUT_FILE = Path(output_csv_path)
    records = []
    with INPUT_FILE.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print('⚠️  Skipping bad JSON:', e)
                continue
            custom_id = obj.get('custom_id', '')
            (idx, gender, ethnicity) = parse_custom_id(custom_id)
            decision = extract_decision(obj)
            records.append({'index': idx, 'gender': gender, 'ethnicity': ethnicity, 'decision': decision})
    with OUTPUT_FILE.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'gender', 'ethnicity', 'decision'])
        writer.writeheader()
        writer.writerows(records)
    print(f'✅ Parsed {len(records)} records → {OUTPUT_FILE}')
