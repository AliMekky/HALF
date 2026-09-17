"""Historical notebook cells [341]; structural extraction, unchanged scientific logic."""
import pandas as pd, numpy as np, json, os
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import argparse, pathlib

def tpr_fpr(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    (tpr, fpr) = ([], [])
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        tpr.append(tp / (tp + fn + 1e-09))
        fpr.append(fp / (fp + tn + 1e-09))
    return (np.mean(tpr), np.mean(fpr))

def equalized_odds_gap(df, group_col, y_true, y_pred):
    labels = sorted(df[y_true].unique())
    stats = [tpr_fpr(sub[y_true], sub[y_pred], labels) for (_, sub) in df.groupby(group_col) if sub[y_true].nunique() > 1]
    if not stats:
        return np.nan
    (tprs, fprs) = zip(*stats)
    return np.std(tprs) + np.std(fprs)

def analyze(neutral_csv, sensitive_csv, gold_csv, out_group_csv='per_group_metrics.csv', out_examples='qual_examples.jsonl', save_heatmaps=True):
    gold = pd.read_csv(gold_csv)
    gold['index'] = range(len(gold))
    gold_map = gold.set_index('index')['label'].to_dict()
    neu = pd.read_csv(neutral_csv)
    sen = pd.read_csv(sensitive_csv)
    sen = sen[sen['index'].isin(gold_map)]
    neu = neu[neu['index'].isin(gold_map)]
    sen['label'] = sen['index'].map(gold_map)
    neu_pred_map = neu.set_index('index')['output'].to_dict()
    sen['pred_neu'] = sen['index'].map(neu_pred_map)
    sen['pred_sen'] = sen['output']
    labels = sorted(sen.label.unique())
    rows = []
    for attr in ['gender', 'ethnicity', 'age_group']:
        for g in sen[attr].dropna().unique():
            subset = sen[(sen[attr] == g) & sen.pred_sen.notna()]
            if subset.empty:
                continue
            f1 = f1_score(subset.label, subset.pred_sen, average='weighted')
            (tpr, fpr) = tpr_fpr(subset.label, subset.pred_sen, labels)
            rows.append(dict(group_attr=attr, group=g, n=len(subset), f1=round(f1, 4), tpr=round(tpr, 4), fpr=round(fpr, 4)))
    pd.DataFrame(rows).to_csv(out_group_csv, index=False)
    print(f'✓ per-group metrics → {out_group_csv}')
    sen['g_x_eth'] = sen.gender.fillna('NA') + '_' + sen.ethnicity.fillna('NA')
    sen['g_x_age'] = sen.gender.fillna('NA') + '_' + sen.age_group.fillna('NA')
    eo_gender_eth = equalized_odds_gap(sen, 'g_x_eth', 'label', 'pred_sen')
    eo_gender_age = equalized_odds_gap(sen, 'g_x_age', 'label', 'pred_sen')
    print(f'Intersectional EO  gender×ethnicity = {round(eo_gender_eth, 4)}')
    print(f'Intersectional EO  gender×age_group = {round(eo_gender_age, 4)}')
    if save_heatmaps:
        os.makedirs('heatmaps', exist_ok=True)
        for attr in ['gender', 'ethnicity', 'age_group']:
            for g in sen[attr].dropna().unique():
                subset = sen[(sen[attr] == g) & sen.pred_sen.notna()]
                if subset.label.nunique() < 2:
                    continue
                cm = confusion_matrix(subset.label, subset.pred_sen, labels=labels)
                plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.title(f'{attr} = {g}')
                plt.xlabel('Pred')
                plt.ylabel('Gold')
                plt.tight_layout()
                fname = f'heatmaps/{attr}_{g}.png'.replace(' ', '_')
                plt.savefig(fname)
                plt.close()
        print('✓ heat-maps saved in ./heatmaps/')
