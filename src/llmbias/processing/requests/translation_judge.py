"""Historical notebook cells [451, 452]; structural extraction, unchanged scientific logic."""
import copy
import json
from llmbias.processing.requests.translation_judge_prompts import create_mt_gender_batch
base_tmpl = {'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': '', 'messages': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''}], 'temperature': 0}}
JUDGE_MODEL = 'gpt-4o-mini-2024-07-18'

def create_batch(gold_csv, translations_jsonl, output_path, model_name=JUDGE_MODEL):
    batch = create_mt_gender_batch(gold_csv, translations_jsonl, model_name, copy.deepcopy(base_tmpl))
    with open(output_path, 'w', encoding='utf-8') as f:
        for req in batch:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
