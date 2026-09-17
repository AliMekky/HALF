"""Historical notebook cells [300]; structural extraction, unchanged scientific logic."""
import json
import csv
from pathlib import Path

def parse_custom_id(custom_id: str):
    """
    Extract (index, gender, ethnicity) from the hyphen-delimited custom_id.
    Pattern assumed:  request-<idx>-<…>-<gender>-<ethnicity>
    """
    parts = custom_id.split('-')
    if len(parts) < 3:
        return (None, None, None)
    idx = parts[1]
    gender = parts[-2]
    ethnicity = parts[-1]
    try:
        idx = int(idx)
    except ValueError:
        pass
    return (idx, gender, ethnicity)

def extract_decision(obj: dict):
    """
    Return the model's 'Admit' / 'Reject' string (title-cased).
    """
    try:
        txt = obj['result']['message']['content'][0]['text']
        return txt.strip().title()
    except (KeyError, IndexError, TypeError):
        return None

def process_file(input_path, output_csv_path):
    INPUT_FILE = Path(input_path)
    OUTPUT_FILE = Path(output_csv_path)
    records = []
    buffer = []
    depth = 0
    with INPUT_FILE.open(encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            depth += stripped.count('{') - stripped.count('}')
            buffer.append(line)
            if depth == 0:
                raw = ''.join(buffer)
                buffer.clear()
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    print('⚠️  Skipping malformed object:', e)
                    continue
                cid = obj.get('custom_id', '')
                (idx, g, e_) = parse_custom_id(cid)
                decision = extract_decision(obj)
                records.append({'index': idx, 'gender': g, 'ethnicity': e_, 'decision': decision})
    with OUTPUT_FILE.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'gender', 'ethnicity', 'decision'])
        writer.writeheader()
        writer.writerows(records)
    print(f'✅ Parsed {len(records)} objects → {OUTPUT_FILE}')
