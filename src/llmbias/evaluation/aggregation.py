"""Reconstruction of paper Equation (1), from already normalized dataset scores.

This is new formula-based code, not recovered experiment code. It does not
implement Appendix D normalization from raw metrics. Summarization has weight 1,
consistent with the prose and all eight displayed Table 2 totals.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {
    'medbullets': 3, 'medical_bias': 3, 'ecthr': 3, 'CAMS': 3, 'SAD': 3, 'djinni': 3,
    'education_ranking': 2, 'mt_gender': 2, 'ontonotes': 1, 'bold': 1, 'bbq': 1,
}


def aggregate(scores):
    """Score each model over its nonmissing datasets, as D(m) in Equation (1)."""
    if set(scores.columns) != {'model', *WEIGHTS}:
        raise ValueError('Expected model and exactly the eleven Table 2 dataset columns')
    if scores['model'].isna().any() or scores['model'].duplicated().any():
        raise ValueError('Model identifiers must be present and unique')
    values = scores[list(WEIGHTS)].apply(pd.to_numeric, errors='raise')
    observed = values.notna()
    if ((values < 0) | (values > 1)).any().any():
        raise ValueError('Normalized scores must be finite values between 0 and 1')
    if not np.isfinite(values.fillna(0).to_numpy()).all():
        raise ValueError('Normalized scores must be finite')
    counts = observed.sum(axis=1)
    if (counts == 0).any():
        raise ValueError('Each model needs at least one observed dataset score')
    weights = pd.Series(WEIGHTS)
    total_weight = observed.mul(weights).sum(axis=1)
    return pd.DataFrame({
        'model': scores['model'],
        'naive': values.mean(axis=1) * 100,
        'half': values.mul(weights).sum(axis=1) / total_weight * 100,
        'datasets': counts,
        'total_weight': total_weight,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(pd.read_csv(args.scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        result.to_csv(stream, index=False)


if __name__ == '__main__':
    main()
