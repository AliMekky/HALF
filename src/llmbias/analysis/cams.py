"""Functions extracted from notebook cell 338 (zero-based); legacy globals may be required."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
def equalized_odds_gap(df, group_col, gold_col, pred_col):
    labels = sorted(df[gold_col].unique())
    if len(labels) < 2:
        return np.nan

    tprs, fprs = [], []
    for _, sub in df.groupby(group_col):
        if sub[gold_col].nunique() < 2:
            continue
        cm = confusion_matrix(sub[gold_col], sub[pred_col], labels=labels)
        cls_tpr, cls_fpr = [], []
        for i in range(len(labels)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            cls_tpr.append(tp / (tp + fn + 1e-9))
            cls_fpr.append(fp / (fp + tn + 1e-9))
        tprs.append(np.mean(cls_tpr))
        fprs.append(np.mean(cls_fpr))

    return np.std(tprs) + np.std(fprs) if tprs else np.nan
def load_gold_labels(path):
    df = pd.read_csv(path)
    df["index"] = range(len(df))
    return df.set_index("index")["label"]
def main(gold_path, neutral_path, sensitive_path, out_path):
    gold_series = load_gold_labels(gold_path)
    gold_index_set = set(gold_series.index)

    # ------------------------- predictions ------------------------------
    neu = pd.read_csv(neutral_path)
    sen = pd.read_csv(sensitive_path)

    # Keep rows whose index exists in the gold file
    neu = neu[neu["index"].isin(gold_index_set)]
    sen = sen[sen["index"].isin(gold_index_set)]

    # Attach gold label & predictions
    neu["label"]    = neu["index"].map(gold_series)
    neu_pred_map    = neu.set_index("index")["output"].to_dict()

    sen["label"]    = sen["index"].map(gold_series)
    sen["pred_neu"] = sen["index"].map(neu_pred_map)    # NaN if missing
    sen["pred_sen"] = sen["output"]

    # Drop rows lacking either prediction
    before = len(sen)
    sen = sen.dropna(subset=["pred_neu", "pred_sen"])
    dropped = before - len(sen)
    if dropped:
        print(f"⚠  Dropped {dropped} rows without predictions.")

    # -------------------------- metrics ---------------------------------
    f1_neu = f1_score(sen["label"], sen["pred_neu"], average="weighted")
    f1_sen = f1_score(sen["label"], sen["pred_sen"], average="weighted")

    metrics = {
        "gold_csv"      : Path(gold_path).name,
        "neutral_csv"   : Path(neutral_path).name,
        "sensitive_csv" : Path(sensitive_path).name,
        "f1_neutral"    : round(f1_neu, 4),
        "f1_sensitive"  : round(f1_sen, 4),
        "delta_f1"      : round(f1_sen - f1_neu, 4),
    }

    for attr in ["gender", "ethnicity", "age_group"]:
        eo_neu = equalized_odds_gap(sen, attr, "label", "pred_neu")
        eo_sen = equalized_odds_gap(sen, attr, "label", "pred_sen")
        metrics[f"eo_{attr}_neutral"] = round(eo_neu, 4) if pd.notna(eo_neu) else ""
        metrics[f"eo_{attr}_sensitive"] = round(eo_sen, 4) if pd.notna(eo_sen) else ""
        metrics[f"delta_eo_{attr}"] = (
            round(eo_sen - eo_neu, 4)
            if pd.notna(eo_neu) and pd.notna(eo_sen) else ""
        )

    # -------------------------- save / append --------------------------
    out_file = Path(out_path)
    pd.DataFrame([metrics]).to_csv(
        out_file, mode="a", index=False, header=not out_file.exists()
    )
    print(f"✓  Appended metrics to {out_file.resolve()}")
