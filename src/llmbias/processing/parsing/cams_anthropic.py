"""Historical notebook cells [330]; structural extraction, unchanged scientific logic."""
import json
import re
import argparse
import csv

def extract_output(text):
    """Extracts the numeric output label from the response text."""
    match = re.search('OUTPUT:\\s*(\\d+)', text)
    return match.group(1) if match else None

def parse_custom_id(custom_id):
    """
    Parses the custom_id into:
    - category (neutral or modified)
    - index
    - optionally: ethnicity, gender, age_group
    """
    parts = custom_id.split('-')
    index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    if len(parts) > 8:
        ethnicity = parts[-1]
        gender = parts[-3]
        age_group = parts[-2]
        return ('modified', index, ethnicity, gender, age_group)
    else:
        return ('neutral', index, None, None, None)

def process_file(input_path, output_csv_path):
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get('custom_id', '')
            (category, index, ethnicity, gender, age_group) = parse_custom_id(custom_id)
            try:
                content = data['result']['message']['content'][0]['text']
                output = extract_output(content)
            except (KeyError, IndexError, TypeError):
                continue
            if output is None:
                continue
            rows.append({'index': index, 'gender': gender, 'ethnicity': ethnicity, 'age_group': age_group, 'output': output})
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['index', 'gender', 'ethnicity', 'age_group', 'output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'✅ Wrote {len(rows)} records to {output_csv_path}')
