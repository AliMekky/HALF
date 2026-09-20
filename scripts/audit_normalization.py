"""Compare reconstructed normalization of PDF reference values with Table 2."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from llmbias.evaluation.normalization import normalize


def audit():
    ref = pd.read_csv(ROOT/'experiments/reference/table_02_scores.csv').set_index('model')
    report = {}
    for name, mode, pooled in [('signed','signed',False), ('absolute','absolute',False),
                               ('pooled_absolute','absolute',True)]:
        filename = 'pooled_metrics.csv' if pooled else 'appendix_d_metrics.csv'
        metrics = pd.read_csv(ROOT/'experiments/reference'/filename)
        calibration = pd.read_csv(ROOT/'experiments/reference/pooled_calibration.csv') if pooled else None
        scores, params = normalize(metrics, models=ref.index.tolist(), bbq_mode=mode, calibration=calibration)
        wide = scores.pivot(index='model', columns='dataset', values='score').reindex(index=ref.index, columns=ref.columns)
        matches = wide.round(2).eq(ref)
        report[name] = {'matching_cells':int(matches.to_numpy().sum()), 'total_cells':88,
            'by_dataset':{k:int(v) for k,v in matches.sum().items()},
            'mismatches':[{'model':model,'dataset':dataset,'reconstructed':float(wide.loc[model,dataset]),
                           'paper':float(ref.loc[model,dataset])}
                          for model in ref.index for dataset in ref.columns if not matches.loc[model,dataset]],
            'parameters':params,
            'evidence_limit':'Inputs transcribed from rounded paper tables, not newly regenerated raw predictions.'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh output file')
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for name, result in report.items():
        print(f'{name}: {result["matching_cells"]}/{result["total_cells"]} displayed cells match')


if __name__ == '__main__': main()
