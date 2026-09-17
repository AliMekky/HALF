"""Historical notebook cells [304]; structural extraction, unchanged scientific logic."""
import pandas as pd
from pathlib import Path

def load_data(model):
    neutral_path = INPUT_DIR / f'{model}_djinni_neutral.csv'
    modified_path = INPUT_DIR / f'{model}_djinni_v2.csv'
    df_neutral = pd.read_csv(neutral_path)
    df_modified = pd.read_csv(modified_path)
    df_neutral['model'] = model
    df_modified['model'] = model
    return (df_neutral, df_modified)

def admit_bool(series):
    return series.str.lower().str.strip() == 'admit'

def evaluate_model(model, df_neutral, df_modified):
    neutral_map = admit_bool(df_neutral.set_index('index')['decision'])
    df_mod = df_modified.copy()
    df_mod['admit'] = admit_bool(df_mod['decision'])
    df_mod['neutral_admit'] = df_mod['index'].map(neutral_map)
    df_mod['flipped'] = df_mod['admit'] != df_mod['neutral_admit']
    grouped = df_mod.groupby(['gender', 'ethnicity']).agg(group_admit_rate=('admit', 'mean'), neutral_ref_rate=('neutral_admit', 'mean'), rate_diff=('admit', lambda x: x.mean() - df_mod['neutral_admit'].mean()), flip_rate=('flipped', 'mean'), count=('index', 'count')).reset_index()
    grouped['relative_rate'] = grouped['group_admit_rate'] / grouped['neutral_ref_rate']
    grouped['model'] = model
    return grouped
