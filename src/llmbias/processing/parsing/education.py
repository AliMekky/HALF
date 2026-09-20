"""Historical notebook cells [364]; structural extraction, unchanged scientific logic."""
import json
import re
import csv

def parse_custom_id(input_path, custom_id):
    """
    Extracts index, role, gender, and ethnicity from custom_id like:
    request-0-gpt-4.1-2025-04-14-education_ranking-teacher-male-western
    """
    if 'deepseek' in input_path or 'claude' in input_path:
        parts = custom_id.split('-')
        if len(parts) < 6:
            return (None, None, None, None)
        index = parts[1]
        role = parts[-3]
        gender = parts[-2]
        ethnicity = parts[-1]
    else:
        parts = custom_id.split('-')
        if len(parts) < 8:
            return (None, None, None, None)
        index = parts[1]
        role = parts[-3]
        gender = parts[-2]
        ethnicity = parts[-1]
    return (index, role, gender, ethnicity)

def extract_answer(input_path, response_obj):
    """Extract the model's answer letter from response JSON."""
    try:
        if 'deepseek' in input_path:
            return response_obj['response']
        elif 'claude' in input_path:
            return response_obj['result']['message']['content'][0]['text'].strip()
        return response_obj['response']['body']['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError, TypeError):
        return None

def process_jsonl(input_path, output_csv_path):
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print('error')
                continue
            custom_id = data.get('custom_id', '')
            (index, role, gender, ethnicity) = parse_custom_id(input_path, custom_id)
            answer = extract_answer(input_path, data)
            if not all([index, role, gender, ethnicity, answer]):
                print(index, role, gender, ethnicity, answer)
                continue
            rows.append({'index': index, 'role': role, 'gender': gender, 'ethnicity': ethnicity, 'answer': answer})
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['index', 'role', 'gender', 'ethnicity', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'✅ Extracted {len(rows)} entries to: {output_csv_path}')
