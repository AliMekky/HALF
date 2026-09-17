"""Historical notebook cells [420]; structural extraction, unchanged scientific logic."""
import pandas as pd
import json
import ast
import re

def fix_numpy_array_strings(val):
    """Clean up NumPy array strings inside a dict string so literal_eval works."""
    if isinstance(val, str):
        val = re.sub('array\\(\\s*(\\[.*?\\])\\s*,\\s*dtype=object\\s*\\)', '\\1', val, flags=re.DOTALL)
        try:
            return ast.literal_eval(val)
        except Exception as e:
            print(f'\n❌ Error parsing: {e}\nFor input:\n{val}\n')
            raise
    return val

def convert_numpy_array_to_list(info_dict):
    return {k: list(v) if not isinstance(v, list) else v for (k, v) in info_dict.items()}

def process_file(input_path, gold_csv, output_path):
    path = str(input_path)
    out_path = str(output_path)
    df = pd.read_csv(gold_csv)
    df['additional_metadata'] = df['additional_metadata'].apply(fix_numpy_array_strings)
    df['answer_info'] = df['answer_info'].apply(fix_numpy_array_strings)
    with open(path) as f:
        preds = [json.loads(line) for line in f]
    pred_map = {}
    for p in preds:
        custom_id = p['custom_id']
        if 'claude_bbq' in path:
            content = p['result']['message']['content'][0]['text']
        elif 'deepseek_chat' in path:
            if 'response' in p:
                content = p['response']
            else:
                content = None
        else:
            content = p['response']['body']['choices'][0]['message']['content'].strip().upper()
        parts = custom_id.split('-')
        example_id = int(parts[-2])
        question_index = parts[-1]
        ans_map = {'A': 'ans0', 'B': 'ans1', 'C': 'ans2'}
        pred_map[example_id, question_index] = ans_map.get(content, 'unknown')
    df['gpt-4.1-2025-04-14'] = df.apply(lambda row: pred_map.get((row['example_id'], str(row['question_index'])), 'unknown'), axis=1)
    df['answer_info'] = df['answer_info'].apply(convert_numpy_array_to_list)
    records = []
    for (_, row) in df.iterrows():
        record = {'example_id': row['example_id'], 'question_index': str(row['question_index']), 'question_polarity': row['question_polarity'], 'context_condition': row['context_condition'], 'category': row['category'], 'answer_info': row['answer_info'], 'additional_metadata': ast.literal_eval(row['additional_metadata']) if isinstance(row['additional_metadata'], str) else row['additional_metadata'], 'context': row['context'], 'question': row['question'], 'ans0': row['ans0'], 'ans1': row['ans1'], 'ans2': row['ans2'], 'label': int(row['label']), 'gpt-4.1-2025-04-14': row['gpt-4.1-2025-04-14']}
        records.append(record)
    with open(out_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
