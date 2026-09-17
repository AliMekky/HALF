"""Historical notebook cells [395]; structural extraction, unchanged scientific logic."""
import json
import argparse
from pathlib import Path
import pandas as pd
SYSTEM_PROMPT = 'You are a data-cleaning assistant.\nTask: Given a legal or journalistic passage, identify the placeholder\ntoken(s) that stand for the main **human** protagonist (e.g., “Mr X”, “the applicant”).\nIgnore placeholders for companies, courts, or secondary people unless no human\nprotagonist exists.\n\nReturn ONLY the exact placeholder string(s), including exact casing and spacing,\n**as they appear in the text**, in a JSON array. No commentary, no extra keys.\nIf none are found, return []. Usually, return a single placeholder unless the\npassage clearly has multiple distinct main protagonists.'
USER_PROMPT = '##########  EXAMPLE 1  ##########\n\nUSER:\n<EXCERPT>\n5.  **The applicant** was born in 1940 and lives in Odesa.  \n… 8.  Starting from April 2001, company Y. no longer allowed access …  \n… 26.  On 3 February 2006 the Supreme Court upheld the decisions.  \n</EXCERPT>\n\nIdentify the placeholder(s) to be replaced.\n\nASSISTANT:\n["the applicant"]\n\n##########  YOUR TURN  ##########\nUSER:\n<EXCERPT>\n{text}\n</EXCERPT>\n\nHint: The protagonist is **{gender}**.\n\nIdentify the placeholder(s) to be replaced.\n'

def create_batch(input_path, output_path):
    requests = []
    df = pd.read_csv(input_path, encoding='utf-8')
    for (i, row) in df.iterrows():
        text = row['text']
        gender = 'male' if row['applicant_gender'] == 1 else 'female'
        req = {'custom_id': f'request-{i}', 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'o4-mini', 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': USER_PROMPT.format(text=text, gender=gender)}]}}
        requests.append(req)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for req in requests:
            f_out.write(json.dumps(req, ensure_ascii=False) + '\n')
    print(f'✅ Batch file saved to: {output_path}')
