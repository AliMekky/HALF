"""Functions extracted from notebook cell 358 (zero-based). Preserved evaluation logic."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
LABELS = ['Financial_Problem', 'Everyday_Decision_Making', 'Emotional_Turmoil', 'School', 'Family_Issues', 'Social_Relationships', 'Work', 'Health_Fatigue_Physical_Pain', 'Other']
def compute_multilabel_f1(y_true, y_pred):
    """Compute micro-averaged F1 score for multi-label binary predictions."""
    return f1_score(y_true, y_pred, average="micro", zero_division=0)
def compute_group_metrics(gold_df, sensitive_df, neutral_df, label_cols, group_attr):
    all_metrics = []

    # Merge neutral predictions by index for aligned comparison
    neutral_map = neutral_df.set_index("index")[label_cols]

    for group_value in sensitive_df[group_attr].dropna().unique():
        subset = sensitive_df[sensitive_df[group_attr] == group_value].copy()

        # Ensure we only evaluate common indices
        subset = subset[subset["index"].isin(neutral_map.index)]
        if subset.empty:
            continue

        y_true = gold_df.loc[subset["index"], LABELS].values
        y_pred_sen = subset[LABELS].values
        y_pred_neu = neutral_df.set_index("index").loc[subset["index"], LABELS].values

        f1_sen = compute_multilabel_f1(y_true, y_pred_sen)
        f1_neu = compute_multilabel_f1(y_true, y_pred_neu)

        all_metrics.append({
            "group_attr": group_attr,
            "group": group_value,
            "f1_sensitive": round(f1_sen, 4),
            "f1_neutral": round(f1_neu, 4),
            "delta_f1": round(f1_sen - f1_neu, 4),
            "n": len(subset)
        })

    return all_metrics
def evaluate_model(gold_csv, neutral_csv, sensitive_csv, label_cols, model_name, output_csv_path):
    gold_df = pd.read_csv(gold_csv)
    if "index" not in gold_df.columns:
        gold_df["index"] = range(len(gold_df))
    gold_df = gold_df.set_index("index")

    neutral_df = pd.read_csv(neutral_csv)
    sensitive_df = pd.read_csv(sensitive_csv)

    all_metrics = []
    for attr in ["gender", "ethnicity", "age_group"]:
        attr_metrics = compute_group_metrics(gold_df, sensitive_df, neutral_df, label_cols, attr)
        for row in attr_metrics:
            row["model"] = model_name
            all_metrics.append(row)

    pd.DataFrame(all_metrics).to_csv(output_csv_path, index=False)
    print(f"✅ Saved: {output_csv_path}")
def merge_all_metrics(per_group_dir, output_csv):
    all_dfs = []
    for file in Path(per_group_dir).glob("*_SAD_metrics.csv"):
        df = pd.read_csv(file)
        all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df[["model", "group_attr", "group", "f1_neutral", "f1_sensitive", "delta_f1", "n"]]
        final_df.to_csv(output_csv, index=False)
        print(f"📊 Final SAD summary saved to: {output_csv}")
    else:
        print("⚠️ No *_SAD_metrics.csv files found.")
