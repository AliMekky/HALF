"""Functions extracted from notebook cell 343 (zero-based); legacy globals may be required."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
def tpr_fpr(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tpr, fpr = [], []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        tpr.append(tp / (tp + fn + 1e-9))
        fpr.append(fp / (fp + tn + 1e-9))
    return np.mean(tpr), np.mean(fpr)
def compute_all_group_metrics(gold_csv, model_configs):
    gold = pd.read_csv(gold_csv)
    gold["index"] = range(len(gold))
    gold_map = gold.set_index("index")["label"].to_dict()
    results = []

    for config in model_configs:
        model_name = config["name"]
        sen = pd.read_csv(config["sensitive"])
        neu = pd.read_csv(config["neutral"])

        
        sen = sen[sen["index"].isin(gold_map)]
        neu = neu[neu["index"].isin(gold_map)]
        
        sen["label"] = sen["index"].map(gold_map)
        neu_pred_map = neu.set_index("index")["output"].to_dict()
        sen["pred_neu"] = sen["index"].map(neu_pred_map)
        sen["pred_sen"] = sen["output"]

        labels = sorted(sen.label.unique())
        for attr in ["gender", "ethnicity", "age_group"]:
            for g in sen[attr].dropna().unique():
                subset = sen[(sen[attr] == g) & sen.pred_sen.notna() & sen.pred_neu.notna()]
                if subset.empty:
                    print(model_name)
                    continue
                print(f"Processing {attr} = {g} with {len(subset)} samples")
                f1_sen = f1_score(subset.label, subset.pred_sen, average="weighted")
                f1_neu = f1_score(subset.label, subset.pred_neu, average="weighted")
                delta_f1 = f1_sen - f1_neu
                tpr, fpr = tpr_fpr(subset.label, subset.pred_sen, labels)
                results.append({
                    "model": model_name,
                    "group_attr": attr,
                    "group": g,
                    "f1_sensitive": round(f1_sen, 4),
                    "f1_neutral": round(f1_neu, 4),
                    "delta_f1": round(delta_f1, 4),
                    "tpr": round(tpr, 4),
                    "fpr": round(fpr, 4),
                    "n": len(subset)
                })

    return pd.DataFrame(results)
