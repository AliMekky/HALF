"""Historical notebook cells [386]; structural extraction, unchanged scientific logic."""
import json
import argparse
from pathlib import Path
import pandas as pd
SYSTEM_PROMPT = 'You are a legal-text expert.\nGiven a passage from an ECtHR judgment, infer the gender of “the applicant”.\nReply with **only one digit** and please do not add any additional text or explanation:\n    0 → unknown / not clear\n    1 → male\n    2 → female'

def create_batch(input_path, output_path):
    requests = []
    df = pd.read_csv(input_path, encoding='utf-8')
    for (i, row) in df.iterrows():
        text = row['text']
        if row['applicant_gender'] != 0:
            continue
        req = {'custom_id': f'request-{i}', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'o4-mini', 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': text}]}}
        requests.append(req)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for req in requests:
            f_out.write(json.dumps(req, ensure_ascii=False) + '\n')
    print(f'✅ Batch file saved to: {output_path}')
