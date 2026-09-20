"""Historical notebook cells [412]; structural extraction, unchanged scientific logic."""
import argparse, json, pathlib, re, sys
from collections import defaultdict
from typing import Dict, Tuple
import pathlib

def load_dataset(path: pathlib.Path) -> Dict[Tuple[str, int, int], dict]:
    """
    Build a lookup keyed by (article_id, pair_id, sample_id) → instance_dict
    so we can copy `text` and `instructions`.
    """
    lookup = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = (obj['article_id'], obj['pair_id'])
            lookup[key] = obj
    return lookup

def extract_ids(custom_id: str) -> Tuple[str, int, int]:
    """
    Parse the OpenAI custom_id string.

    Expected pattern:
      request-{sample}-{model}-...-{article_id}-{pair}
      e.g. request-0-gpt-4.1-2025-04-14-ontonotes-wsj_0347-1
    """
    m_sample = re.match('request-(\\d+)-', custom_id)
    if not m_sample:
        raise ValueError(f'Cannot parse sample_id from {custom_id}')
    sample_id = int(m_sample.group(1))
    parts = custom_id.split('-')
    article_id = parts[-2]
    pair_id = int(parts[-1])
    return (article_id, pair_id)

def convert(pred_path: pathlib.Path, dataset_lookup: Dict[Tuple[str, int, int], dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (n_converted, n_skipped) = (0, 0)
    with pred_path.open(encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for raw in fin:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            try:
                (article_id, pair_id) = extract_ids(obj['custom_id'])
            except ValueError as e:
                print('! skip line:', e, file=sys.stderr)
                n_skipped += 1
                continue
            key = (article_id, pair_id)
            if key not in dataset_lookup:
                print(f'! no source instance for {key} – skipped', file=sys.stderr)
                n_skipped += 1
                continue
            src = dataset_lookup[key]
            if 'deepseek' in str(pred_path):
                if 'response' in obj:
                    summary = obj['response']
                else:
                    print(obj)
                    summary = ''
            elif 'claude' in str(pred_path):
                summary = obj['result']['message']['content'][0]['text'].strip()
            else:
                summary = obj['response']['body']['choices'][0]['message']['content'].strip()
            new_obj = {'article_id': article_id, 'pair_id': pair_id, 'sample_id': 0, 'text': src['text'], 'summary': summary, 'instructions': src['instructions'], 'parse': None}
            fout.write(json.dumps(new_obj, ensure_ascii=False) + '\n')
            n_converted += 1
    print(f'✓ converted: {n_converted} lines  •  skipped: {n_skipped}')
    print('→ written to', out_path)

def process_file(input_path, dataset_path, output_path):
    convert(pathlib.Path(input_path), load_dataset(pathlib.Path(dataset_path)), pathlib.Path(output_path))
