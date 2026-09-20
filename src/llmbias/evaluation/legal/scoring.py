"""
ECtHR fairness table (mF1, LD_KL, WCI, counts)
------------------------------------------------
Encodings expected (FairLex):
    defendant_state : 0 = E.C. European , 1 = The Rest
    applicant_gender: 0 = n/a , 1 = Male , 2 = Female
    applicant_age   : 0 = n/a , 1 = ≤35 , 2 = ≤65 , 3 = >65
"""

import argparse, json, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from scipy.stats import entropy

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def pct_fmt(n, total):                 # "7 224 (80%)"
    return f"{n:,} ({int(round(100*n/total))}%)"

def group_ldkl(mask):
    """KL(label-distribution in this group ‖ overall)"""
    pg = y_true[mask].sum(0)
    if pg.sum() == 0:
        return 0.0
    pg = pg / pg.sum()
    return float(entropy(pg, overall_dist))

def add_rows(axis_values, mapping, drop_val=None):
    """Return table rows for one bias axis."""
    global_rows = []
    valid_mask = np.ones(len(axis_values), bool)
    if drop_val is not None:
        valid_mask &= (axis_values != drop_val)
    total = valid_mask.sum()

    for code, name in mapping.items():
        if code == drop_val:
            continue
        mask = valid_mask & (axis_values == code)
        if mask.sum() == 0:
            continue

        g_f1   = f1_score(y_true[mask], y_pred[mask], average="macro")
        ldkl   = round(group_ldkl(mask), 2)
        wci    = round(abs(g_f1 - overall_macro_F1), 2)

        global_rows.append({
            "Group": name,
            "mF1": round(g_f1 * 100, 1),
            "#train-cases (%)": pct_fmt(int(mask.sum()), total),
            "LD_KL": ldkl,
            "WCI":   wci
        })
    return global_rows
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="file prefix for .pt tensors")
    ap.add_argument("--csv",    required=True, help="original metadata csv (for counts)")
    args = ap.parse_args()

    # ---------- load tensors ----------
    y_true = torch.load(f"{args.prefix}_y_true.pt", map_location="cpu").numpy()
    y_pred = torch.load(f"{args.prefix}_y_pred.pt", map_location="cpu").numpy()
    meta   = torch.load(f"{args.prefix}_metadata.pt", map_location="cpu").numpy()

    overall_macro_F1 = f1_score(y_true, y_pred, average="macro")
    overall_dist     = y_true.sum(0) / y_true.sum()

    # ---------- meta columns ----------
    state, age, gender = meta[:,0], meta[:,1], meta[:,2]

    STATE_MAP  = {0: "E.C. European", 1: "The Rest"}
    GENDER_MAP = {1: "Male", 2: "Female"}            # drop 0 (n/a)
    AGE_MAP    = {1: "≤ 35 years", 2: "≤ 65 years", 3: "> 65 years"}  # drop 0

    table = []
    table += add_rows(state,  STATE_MAP)                     # keep all
    table += add_rows(gender, GENDER_MAP, drop_val=0)        # drop n/a gender
    table += add_rows(age,    AGE_MAP,    drop_val=0)        # drop n/a age

    # ---------- pretty print ----------
    print("\n==== ECtHR Fairness Table ====\n")
    for r in table:
        print(f"{r['Group']:<15}  "
              f"mF1 {r['mF1']:>5}  "
              f"{r['#train-cases (%)']:<12}  "
              f"LD_KL {r['LD_KL']:.2f}  "
              f"WCI {r['WCI']:.2f}")

    # ---------- save ----------
    df = pd.DataFrame(table)
    df.to_csv (f"{args.prefix}_fairness_table.csv",  index=False)
    with open(f"{args.prefix}_fairness_table.json", "w") as f:
        json.dump(table, f, indent=2)

    print(f"\n✓ Saved: {args.prefix}_fairness_table.csv / .json")

