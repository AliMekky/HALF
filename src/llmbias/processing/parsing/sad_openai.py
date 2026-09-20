"""Historical notebook cells [349]; structural extraction, unchanged scientific logic."""
import json
import re
import argparse
import csv
LABELS = ['Financial_Problem', 'Everyday_Decision_Making', 'Emotional_Turmoil', 'School', 'Family_Issues', 'Social_Relationships', 'Work', 'Health_Fatigue_Physical_Pain', 'Other']
LABEL_TO_INDEX = {label.replace(' ', '_'): i for (i, label) in enumerate(LABELS)}

def extract_labels_indices(text):
    """Extract comma-separated indices for active labels."""
    matches = re.findall('([A-Za-z_ ]+):\\s*([01])', text)
    active_labels = []
    for (label, val) in matches:
        label_key = label.strip().replace(' ', '_')
        if val == '1' and label_key in LABEL_TO_INDEX:
            active_labels.append(str(LABEL_TO_INDEX[label_key]))
    return ','.join(sorted(active_labels, key=int))

def extract_labels_onehot(text):
    """Extract one-hot encoded vector for active labels."""
    onehot = [0] * len(LABELS)
    matches = re.findall('([A-Za-z_ ]+):\\s*([01])', text)
    for (label, val) in matches:
        label_key = label.strip().replace(' ', '_')
        if val == '1' and label_key in LABEL_TO_INDEX:
            onehot[LABEL_TO_INDEX[label_key]] = 1
    return onehot

def parse_custom_id(custom_id):
    parts = custom_id.split('-')
    index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    if len(parts) > 8:
        gender = parts[-3]
        age_group = parts[-2]
        ethnicity = parts[-1]
        return ('modified', index, gender, age_group, ethnicity)
    else:
        return ('neutral', index, '', '', '')

def process_file(input_path, output_csv_path, output_format):
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get('custom_id', '')
            (category, index, gender, age_group, ethnicity) = parse_custom_id(custom_id)
            try:
                content = data['response']['body']['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                continue
            if output_format == 'indices':
                output = extract_labels_indices(content)
                if not output:
                    continue
                row = {'index': index, 'gender': gender, 'ethnicity': ethnicity, 'age_group': age_group, 'output': output}
            elif output_format == 'onehot':
                onehot = extract_labels_onehot(content)
                row = {'index': index, 'gender': gender, 'ethnicity': ethnicity, 'age_group': age_group}
                for (i, val) in enumerate(onehot):
                    row[LABELS[i].replace(' ', '_')] = val
            else:
                raise ValueError("Invalid format. Choose 'indices' or 'onehot'.")
            rows.append(row)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if output_format == 'indices':
            fieldnames = ['index', 'gender', 'ethnicity', 'age_group', 'output']
        else:
            fieldnames = ['index', 'gender', 'ethnicity', 'age_group'] + [label.replace(' ', '_') for label in LABELS]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'✅ Wrote {len(rows)} records to {output_csv_path}')
