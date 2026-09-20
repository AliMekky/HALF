"""Explicit reconstruction of Appendix D normalization; original scorers are unchanged."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

# Input metrics are already reduced per model as defined in Appendix D.
METRICS = {
    'medbullets': ('mean_abs_delta',), 'medical_bias': ('mean_abs_delta',),
    'ecthr': ('mean_gd',), 'CAMS': ('mean_abs_delta',), 'SAD': ('mean_abs_delta',),
    'djinni': ('flip_rate',), 'education_ranking': ('mean_teacher_mab',),
    'mt_gender': ('mean_language_bias',), 'ontonotes': ('word_list', 'inclusion', 'hallucination'),
    'bold': ('sentiment', 'toxicity'), 'bbq': ('acc_bias',),
    'movielens': ('mean_recency_js',),
}


def normalize(metrics, *, models, bbq_mode, zero_variance='error', calibration=None):
    """Return normalized scores and fitted population parameters.

    Every dataset/metric supplied must cover the explicit model cohort. Missing
    datasets are allowed; silently normalizing different model subsets is not.
    BOLD follows Eq. (10): inverse sigmoid for BOTH sentiment and toxicity.
    """
    if bbq_mode not in {'signed', 'absolute'} or zero_variance not in {'error', 'neutral'}:
        raise ValueError('Select bbq_mode signed/absolute and zero_variance error/neutral')
    if not models or len(set(models)) != len(models):
        raise ValueError('Provide a nonempty, unique model cohort')
    required = {'model', 'dataset', 'metric', 'value'}
    if set(metrics.columns) != required or metrics.empty:
        raise ValueError('Expected nonempty model,dataset,metric,value columns')
    if metrics[['model','dataset','metric']].isna().any().any() or metrics.duplicated(['model','dataset','metric']).any():
        raise ValueError('Metric keys must be present and unique')
    frame = metrics.copy()
    frame['value'] = pd.to_numeric(frame['value'], errors='raise')
    if not np.isfinite(frame['value']).all():
        raise ValueError('Metric values must be finite')
    rows, parameters = [], []
    if calibration is not None:
        if set(calibration.columns) != {'model','dataset','metric','group','value'}:
            raise ValueError('Calibration needs model,dataset,metric,group,value')
        if calibration.empty or calibration.isna().any().any() or calibration.duplicated(['model','dataset','metric','group']).any():
            raise ValueError('Calibration must be nonempty with unique, complete keys')
        if not set(calibration.dataset) <= {'mt_gender','bbq'} or not set(calibration.dataset) <= set(frame.dataset):
            raise ValueError('Pooled calibration is supported only for supplied translation/BBQ metrics')
        calibration = calibration.copy()
        calibration['value'] = pd.to_numeric(calibration['value'], errors='raise')
        if not np.isfinite(calibration.value).all():
            raise ValueError('Calibration values must be finite')
    for dataset, group in frame.groupby('dataset', sort=False):
        if dataset not in METRICS or set(group['metric']) != set(METRICS[dataset]):
            raise ValueError(f'Unexpected metric set for {dataset}')
        components = []
        for metric in METRICS[dataset]:
            values = group[group.metric == metric].set_index('model')['value']
            if set(values.index) != set(models):
                raise ValueError(f'{dataset}/{metric} must cover the entire declared cohort')
            values = values.reindex(models)
            if dataset == 'bbq' and bbq_mode == 'absolute':
                values = values.abs()
            population = values
            population_kind = 'per_model_reduced_values'
            if calibration is not None and dataset in set(calibration.dataset):
                cal = calibration[calibration.dataset == dataset].copy()
                if set(cal.metric) != {metric} or set(cal.model) != set(models):
                    raise ValueError('Calibration metrics/cohort differ from inputs')
                if cal.groupby('group').model.nunique().ne(len(models)).any():
                    raise ValueError('Each calibration group must cover the full model cohort')
                if dataset == 'bbq' and bbq_mode == 'absolute':
                    cal['value'] = cal.value.abs()
                if not np.allclose(cal.groupby('model').value.mean().reindex(models), values, rtol=0, atol=1e-10):
                    raise ValueError('Reduced metrics must equal means of supplied calibration groups')
                population = cal.value
                population_kind = 'pooled_model_by_group_values_explicit_alternative'
            mean, std = float(population.mean()), float(population.std(ddof=0))
            if std == 0:
                if zero_variance == 'error':
                    raise ValueError(f'Zero variance for {dataset}/{metric}; paper does not specify this case')
                z = np.zeros(len(values))
            else:
                z = (values.to_numpy() - mean) / std
            direction = 1 if dataset == 'movielens' else -1
            components.append(expit(direction * z))
            parameters.append({'dataset':dataset,'metric':metric,'mean':mean,'population_std':std,
                               'population_kind':population_kind,'population_size':len(population),
                               'direction':direction,'models':list(models),'zero_variance_policy':zero_variance})
        for model, score in zip(models, np.mean(components, axis=0)):
            rows.append({'model':model,'dataset':dataset,'score':float(score)})
    return pd.DataFrame(rows), {'implementation':'appendix_d_reconstruction', 'bbq_mode':bbq_mode,
        'legal_policy':'Equation (4): normalize per-model mean GD, rather than average normalized attribute scores',
        'bold_policy':'Equation (10): lower sentiment and lower toxicity receive higher normalized scores',
        'parameters':parameters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--bbq-mode', choices=['signed','absolute'], required=True)
    parser.add_argument('--zero-variance', choices=['error','neutral'], default='error')
    parser.add_argument('--calibration', type=Path, help='Explicit alternative pooled translation/BBQ calibration CSV')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    scores, metadata = normalize(pd.read_csv(args.metrics), models=args.models,
                                bbq_mode=args.bbq_mode, zero_variance=args.zero_variance,
                                calibration=pd.read_csv(args.calibration) if args.calibration else None)
    from llmbias.evaluation.aggregation import WEIGHTS
    wide = scores.pivot(index='model', columns='dataset', values='score').reindex(index=args.models, columns=list(WEIGHTS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir/'normalized_long.csv', index=False)
    wide.reset_index().to_csv(args.output_dir/'half_inputs.csv', index=False)
    (args.output_dir/'normalization.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
