"""Historical notebook cells [382]; structural extraction, unchanged scientific logic."""
from pathlib import Path
from collections import defaultdict
import pandas as pd
from llmbias.evaluation.recommendation.metrics import load_jsonl, req_idx_of, query_type_of, group_of, extract_items, split_attrs, jaccard, prag

def evaluate(data_dir, output_dir):
    DATA_DIR = Path(data_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    neutral = defaultdict(lambda : defaultdict(dict))
    sensitive = defaultdict(list)
    failures = {}
    refusals = {}
    for (fix, fp) in enumerate(DATA_DIR.glob('*movielens*.jsonl')):
        model = fp.name.split('_movielens')[0]
        failures[model] = []
        refusals[model] = []
        is_neu = 'neutral' in fp.name
        print(fp)
        for (i, obj) in enumerate(load_jsonl(fp)):
            cid = obj['custom_id']
            rid = req_idx_of(cid)
            if rid is None:
                continue
            qtype = query_type_of(cid)
            if 'deepseek_chat' in str(fp):
                text = obj['response']
            elif 'claude' in str(fp):
                text = obj['result']['message']['content'][0]['text']
            else:
                try:
                    text = obj['response']['body']['choices'][0]['message']['content']
                except Exception as e:
                    continue
            items = extract_items(text)
            if 'I cannot' in text:
                refusals[model].append(cid)
            if '8B' in str(fp):
                print((i, items))
            if is_neu:
                neutral[model][qtype][rid] = items
            else:
                sensitive[model, qtype, rid].append((group_of(cid), items))
    print(sensitive['o4-mini', 'recent', '0'])
    print(neutral['o4-mini']['recent'])
    for (k, v) in failures.items():
        if v:
            print(f'Model {k} had {len(v)} failures:')
            for (idx, items) in v:
                print(f'  Index {idx}: {items}')
    records = []
    for ((model, qtype, rid), lst) in sensitive.items():
        if rid not in neutral[model][qtype]:
            continue
        n_items = neutral[model][qtype][rid]
        for (group, s_items) in lst:
            (gender, age, eth) = split_attrs(group)
            records.append({'model': model, 'query_type': qtype, 'group': group, 'gender': gender, 'age': age, 'ethnicity': eth, 'JS': jaccard(n_items, s_items), 'PRAG': prag(n_items, s_items)})
    df = pd.DataFrame(records)
    outdir = Path(output_dir)
    df.groupby(['model', 'query_type', 'gender']).agg(JS_mean=('JS', 'mean'), PRAG_mean=('PRAG', 'mean')).reset_index().to_csv(outdir / 'bias_gender_pt2.csv', index=False)
    df.groupby(['model', 'query_type', 'ethnicity']).agg(JS_mean=('JS', 'mean'), PRAG_mean=('PRAG', 'mean')).reset_index().to_csv(outdir / 'bias_ethnicity_pt2.csv', index=False)
    df.groupby(['model', 'query_type', 'age']).agg(JS_mean=('JS', 'mean'), PRAG_mean=('PRAG', 'mean')).reset_index().to_csv(outdir / 'bias_age_pt2.csv', index=False)
    print('✓ Done.  Results saved in:', outdir)
    return df
