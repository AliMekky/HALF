import json
import re
import csv
import argparse

def split_json_objects(raw_text):
    """Split a concatenated JSON string into individual JSON objects."""
    objects = []
    brace_stack = []
    start = None

    for i, char in enumerate(raw_text):
        if char == '{':
            if not brace_stack:
                start = i
            brace_stack.append('{')
        elif char == '}':
            brace_stack.pop()
            if not brace_stack:
                objects.append(raw_text[start:i+1])

    return objects

def parse_custom_id(custom_id):
    match = re.match(r'request-(\d+)-.*-(\w+)-(\w+)', custom_id)
    index = int(custom_id.split('-')[1])
    if match:
        return index, match.group(2), match.group(3)
    return None, None, None

def extract_mcq_data_to_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    json_objects = split_json_objects(raw)
    records = []

    for obj in json_objects:
        try:
            data = json.loads(obj)
        except json.JSONDecodeError:
            continue

        index, ethnicity, gender = parse_custom_id(data.get('custom_id', ''))
        try:
            answer = data['result']['message']['content'][0]['text']
        except (KeyError, IndexError, TypeError):
            answer = None

        records.append({
            'index': index,
            'ethnicity': ethnicity,
            'gender': gender,
            'answer': answer
        })

    # Save to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'ethnicity', 'gender', 'answer'])
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MCQ answers from concatenated JSON objects and save to CSV.")
    parser.add_argument("--input_path", help="Path to input file")
    parser.add_argument("--output_path", help="Path to output CSV")

    args = parser.parse_args()
    extract_mcq_data_to_csv(args.input_path, args.output_path)
