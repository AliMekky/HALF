"""Historical notebook cells [418]; structural extraction, unchanged scientific logic."""
import argparse
import pandas as pd

def summarize_metadata(csv_path):
    df = pd.read_csv(csv_path)
    meta_cols = ['defendant_state', 'applicant_age', 'applicant_gender']
    print(f'✓ Loaded CSV: {csv_path}')
    print(f'Total examples: {len(df)}\n')
    for col in meta_cols:
        print(f'== {col.upper()} ==')
        print(f'Unique values ({df[col].nunique()}):')
        print(df[col].value_counts(dropna=False))
        print(f'Missing: {df[col].isna().sum()}')
        print('-' * 40)
    print('\n== Cross-tab: AGE × GENDER ==')
    print(pd.crosstab(df['applicant_age'], df['applicant_gender'], margins=True))
    print('\n== Cross-tab: STATE × AGE ==')
    print(pd.crosstab(df['defendant_state'], df['applicant_age'], margins=True))
