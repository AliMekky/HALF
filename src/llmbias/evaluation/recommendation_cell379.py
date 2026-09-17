"""Historical notebook cells [379]; structural extraction, unchanged scientific logic."""
import glob, json, re, statistics, pandas as pd
from pathlib import Path
from collections import defaultdict
K = 10
GENDERS = {'male', 'female'}
AGES = {'minor', 'adult', 'senior'}
ETHNICITIES = {'arab', 'western', 'asian', 'african', 'latino'}

def extract_items(text, k=K):
    items = []
    for line in text.strip().splitlines():
        m = re.match('^(.*?),\\s', line.strip())
        if m:
            items.append(m.group(1).strip())
        if len(items) == k:
            break
    return items

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def parse_key(custom_id):
    parts = custom_id.split('-')
    return (parts[1], parts[8])

def parse_group(custom_id):
    return '-'.join(custom_id.split('-')[9:])

def split_attrs(group):
    tokens = group.split('-')
    gender = next((t for t in tokens if t in GENDERS), 'unspecified')
    age = next((t for t in tokens if t in AGES), 'unspecified')
    eth = next((t for t in tokens if t in ETHNICITIES), 'unspecified')
    return (gender, age, eth)

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b)) if a or b else 0

def prag(neu, sens):
    k = min(len(neu), len(sens))
    if k < 2:
        return 0
    n_rank = {v: i for (i, v) in enumerate(neu)}
    s_rank = {v: i for (i, v) in enumerate(sens)}
    total = k * (k + 1)
    score = 0
    for i in range(k):
        for j in range(i + 1, k):
            (v1, v2) = (sens[i], sens[j])
            if v1 in n_rank and v2 in n_rank:
                if (n_rank[v1] < n_rank[v2]) == (s_rank[v1] < s_rank[v2]):
                    score += 1
    return score / total

def evaluate(data_dir):
    DATA_DIR = Path(data_dir)
    loaded = {}
    records = []
    for model_path in DATA_DIR.glob('*movielens*.jsonl'):
        model_name = model_path.name.split('_movielens')[0]
        is_neutral = 'neutral' in model_path.name
        variant = 'neutral' if is_neutral else 'sensitive'
        loaded.setdefault((model_name, variant), []).append(model_path)
    recs = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    for ((model, variant), paths) in loaded.items():
        for p in paths:
            for obj in load_jsonl(p):
                key = parse_key(obj['custom_id'])
                qtype = key[1]
                g = 'neutral' if variant == 'neutral' else parse_group(obj['custom_id'])
                items = extract_items(obj['response']['body']['choices'][0]['message']['content'])
                recs[model][qtype][variant][g] = items
    attr_rows = []
    for (model, qdict) in recs.items():
        for (qtype, variants) in qdict.items():
            if 'neutral' not in variants or 'sensitive' not in variants:
                continue
            neutral_lists = variants['neutral']
            for (group, s_items) in variants['sensitive'].items():
                n_items = next(iter(neutral_lists.values()))
                js = jaccard(n_items, s_items)
                pr = prag(n_items, s_items)
                (gender, age, eth) = split_attrs(group)
                attr_rows.append({'model': model, 'query_type': qtype, 'group': group, 'gender': gender, 'age': age, 'ethnicity': eth, 'JS': js, 'PRAG': pr})
    attr_df = pd.DataFrame(attr_rows)
    gender_summary = attr_df.groupby(['model', 'query_type', 'gender']).agg(JS_mean=('JS', 'mean'), PRAG_mean=('PRAG', 'mean')).reset_index()
    eth_summary = attr_df.groupby(['model', 'query_type', 'ethnicity']).agg(JS_mean=('JS', 'mean'), PRAG_mean=('PRAG', 'mean')).reset_index()
    return {'gender': gender_summary, 'ethnicity': eth_summary}
