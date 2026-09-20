"""Compare original saved predictions with paper Tables 6–7; no inference or metric edits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from llmbias.evaluation.mental_health.sad import LABELS

PREFIXES = ['claude', 'gpt-4.1-2025-04-14', 'gpt-4.1-mini-2025-04-14',
            'o4-mini-2025-04-16', 'deepseek_chat', 'Llama-3.2-1B-Instruct-v2',
            'Llama-3.2-3B-Instruct-v2', 'Meta-Llama-3.1-8B-Instruct-v2']


def audit(source):
    reference = json.loads((ROOT / 'experiments/reference/mental_health_tables.json').read_text())
    report = {'method': 'Recompute weighted, macro, and micro F1 on the original paired rows. '
                        'Compare deltas after the original 4-decimal rounding, at PDF precision. '
                        'This comparison does not modify the published evaluators.',
              'inputs': [], 'datasets': {}}
    def read(path):
        report['inputs'].append({'path':str(path.relative_to(source)),
                                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return pd.read_csv(path)
    for dataset in ['CAMS', 'SAD']:
        gold = read(source / 'mental_health_data' / f'{dataset}.csv')
        if 'index' not in gold:
            gold['index'] = range(len(gold))
        gold = gold.set_index('index')
        results = []
        for mi, (model, prefix) in enumerate(zip(reference['model_order'], PREFIXES)):
            directory = source / 'experiments/mental_health_data/output_files'
            neutral = read(directory / f'{prefix}_{dataset}_neutral.csv').set_index('index')
            sensitive = read(directory / f'{prefix}_{dataset}.csv')
            sensitive = sensitive[sensitive['index'].isin(gold.index) & sensitive['index'].isin(neutral.index)]
            columns = ['output'] if dataset == 'CAMS' else LABELS
            paired = sensitive['index']
            missing = neutral.loc[paired,columns].isna().any(axis=1).to_numpy() | sensitive[columns].isna().any(axis=1).to_numpy()
            sensitive = sensitive.loc[~missing].copy()
            entry = {'model':model,'neutral_rows':len(neutral),'paired_sensitive_rows':len(sensitive),'averages':{}}
            for average in ['weighted','macro','micro']:
                def score(indices, predictions):
                    actual = gold.loc[indices, 'label'].to_numpy() if dataset=='CAMS' else gold.loc[indices,LABELS].to_numpy()
                    pred = predictions['output'].to_numpy() if dataset=='CAMS' else predictions[LABELS].to_numpy()
                    return float(f1_score(actual, pred, average=average, zero_division=0))
                common = neutral.index.intersection(gold.index)
                baseline = score(common, neutral.loc[common]) * 100
                paired_baseline = score(sensitive['index'], neutral.loc[sensitive['index']]) * 100
                expected = reference['values'][dataset]['neutral'][mi]
                groups = []
                for attr in ['gender','ethnicity','age_group']:
                    for group, subset in sensitive.groupby(attr):
                        key = str(group).lower()
                        if key not in reference['values'][dataset]:continue
                        indices=subset['index']
                        neu=score(indices,neutral.loc[indices])
                        sen=score(indices,subset)
                        delta=round(sen-neu,4)*100
                        published=reference['values'][dataset][key][mi]
                        tolerance=.05 if dataset=='CAMS' else .005
                        groups.append({'attribute':attr,'group':key,'n':len(subset),
                                       'neutral_pct':neu*100,'delta_pp':delta,'paper_delta_pp':published,
                                       'matches_paper_precision':abs(delta-published)<=tolerance+1e-9})
                entry['averages'][average]={'neutral_pct':baseline,'paper_neutral_pct':expected,
                    'neutral_matches_paper_precision':abs(baseline-expected)<=.05+1e-9,
                    'paired_neutral_pct':paired_baseline,
                    'paired_neutral_matches_paper_precision':abs(paired_baseline-expected)<=.05+1e-9,
                    'group_matches':sum(x['matches_paper_precision'] for x in groups),'group_count':len(groups),'groups':groups}
            results.append(entry)
        report['datasets'][dataset]=results
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Original LLMBias repository')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a fresh output file')
    result=audit(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for dataset,entries in result['datasets'].items():
        print(dataset)
        for average in ['weighted','macro','micro']:
            print(average,'neutral matches',sum(e['averages'][average]['neutral_matches_paper_precision'] for e in entries),
                  '/ 8; group matches',sum(e['averages'][average]['group_matches'] for e in entries),
                  '/',sum(e['averages'][average]['group_count'] for e in entries))


if __name__=='__main__':main()
