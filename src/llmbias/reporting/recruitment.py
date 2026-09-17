"""Historical notebook cells [308]; structural extraction, unchanged scientific logic."""
import pandas as pd
from pathlib import Path
PALETTE = 'muted'
THRESHOLD = 0.8

def make_plot(subset, group_type, save_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    '\n    Plot relative admit rate vs neutral for a given group type.\n    Saves plot as PNG and displays it.\n    '
    plt.figure(figsize=(10, 5))
    sns.barplot(data=subset, x='group_key', y='relative_rate', hue='model', palette=PALETTE)
    plt.axhline(THRESHOLD, color='red', linestyle='--', label='80% threshold')
    plt.title(f'Relative Admit Rate vs. Neutral – {group_type.capitalize()}')
    plt.ylabel('Relative Rate (group / neutral)')
    plt.xlabel(group_type.capitalize())
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300)
    plt.show()

def report(input_path, output_dir):
    import seaborn as sns
    df = pd.read_csv(input_path)
    sns.set_theme(style='whitegrid')
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for group_type in ['gender_ethnicity', 'gender', 'ethnicity']:
        subset = df[df['group_type'] == group_type].copy()
        make_plot(subset, group_type, str(outdir / f'relative_rate_{group_type}'))
