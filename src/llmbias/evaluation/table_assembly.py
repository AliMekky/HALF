"""Reconstructed reductions for paper tables; require explicit, validated input schemas."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def require(frame, columns, keys, numbers):
    if not set(columns) <= set(frame) or frame.empty:
        raise ValueError(f'Expected nonempty columns: {columns}')
    if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
        raise ValueError(f'Missing or duplicate row keys: {keys}')
    if not np.isfinite(frame[list(numbers)].apply(pd.to_numeric, errors='raise')).all().all():
        raise ValueError('Values must be finite')


def legal(groups):
    """Table 5: equal-weight group mean, population standard deviation, minimum."""
    require(groups, ['model','attribute','group','mf1'], ['model','attribute','group'], ['mf1'])
    if not set(groups.attribute) <= {'state','gender','age'}:
        raise ValueError('Legal attributes must be state, gender, age')
    if not groups.groupby('model').attribute.nunique().eq(3).all():
        raise ValueError('Each model needs all three legal attributes')
    if not groups.mf1.between(0,100).all():
        raise ValueError('Legal mF1 must be expressed as percentages')
    return groups.groupby(['model','attribute'], sort=False).mf1.agg(
        mean_mf1='mean', gd=lambda values: values.std(ddof=0), worst_mf1='min').reset_index()


def recruitment(neutral, groups, *, weighting):
    """Table 8: neutral acceptance from unique inputs; flip rate from one partition.

    Only gender_ethnicity intersection rows are used, avoiding triple-counting
    the evaluator's concatenated gender, ethnicity and intersection summaries.
    """
    require(neutral, ['model','index','decision'], ['model','index'], [])
    if not neutral.decision.astype(str).str.lower().isin(['admit','reject']).all():
        raise ValueError('Neutral decisions must be Admit or Reject')
    if 'group_type' not in groups:
        raise ValueError('Recruitment groups require group_type')
    g = groups[groups.group_type == 'gender_ethnicity'].copy()
    require(g, ['model','group_key','flip_rate','count'], ['model','group_key'], ['flip_rate','count'])
    if not g.flip_rate.between(0,1).all() or not (g['count']>0).all():
        raise ValueError('Flip rates must be fractions and counts positive')
    if set(neutral.model) != set(g.model):
        raise ValueError('Neutral and sensitive model sets differ')
    if weighting not in {'row','equal_group'}:
        raise ValueError('Choose row or equal_group weighting explicitly')
    rows=[]
    for model, subset in g.groupby('model', sort=False):
        flip = np.average(subset.flip_rate, weights=subset['count']) if weighting=='row' else subset.flip_rate.mean()
        n=neutral[neutral.model==model]
        rows.append({'model':model,'neutral_admit_pct':100*n.decision.str.lower().eq('admit').mean(),
                     'flip_pct':100*flip,'weighting':weighting})
    return pd.DataFrame(rows)


def education(cells):
    """Tables 10–11: z-normalize six cell means, then marginalize each dimension.

    Explicit reconstruction: equal-weight gender×ethnicity cells. Do not average
    the old subgroup MDB values; those represent a different quantity.
    """
    require(cells, ['model','role','gender','ethnicity','mean_score'],
            ['model','role','gender','ethnicity'], ['mean_score'])
    if not set(cells.role) <= {'student','teacher'}:
        raise ValueError('Education roles must be student or teacher')
    if not cells.mean_score.between(1,5).all():
        raise ValueError('Education mean scores must use the A–E scale 1–5')
    rows=[]
    for (model, role), group in cells.groupby(['model','role'], sort=False):
        if group.gender.nunique()!=2 or group.ethnicity.nunique()!=3 or len(group)!=6:
            raise ValueError('Each education model/role needs the full 2x3 cell grid')
        std=group.mean_score.std(ddof=0)
        if std==0:raise ValueError('Constant education cell means; z-normalization undefined')
        group=group.assign(z=(group.mean_score-group.mean_score.mean())/std)
        for dimension in ['gender','ethnicity']:
            means=group.groupby(dimension).z.mean()
            rows.append({'model':model,'role':role,'dimension':dimension,
                         'MAB':means.abs().mean(),'MDB':means.max()-means.min()})
    return pd.DataFrame(rows)


def deltas(frame):
    """Tables 3,4,6,7 presentation from caller-verified percentage-point deltas."""
    require(frame, ['model','variant','delta_pp'], ['model','variant'], ['delta_pp'])
    return frame.pivot(index='variant', columns='model', values='delta_pp').reset_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['legal','recruitment','education','deltas'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--neutral', type=Path)
    parser.add_argument('--weighting', choices=['row','equal_group'])
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.input)
    if args.kind=='recruitment':
        if args.neutral is None or args.weighting is None:parser.error('Recruitment needs --neutral and --weighting')
        result=recruitment(pd.read_csv(args.neutral),frame,weighting=args.weighting)
    else:result=globals()[args.kind](frame)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:result.to_csv(f,index=False)


if __name__=='__main__':main()
