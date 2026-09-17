# # # import torch
# # # from fairlex.dataloaders.ecthr_dataset import ECtHRDataset

# # # # ------------  load tensors  -------------
# # # y_pred     = torch.load('ecthr_y_pred.pt')
# # # y_true     = torch.load('ecthr_y_true.pt')
# # # metadata   = torch.load('ecthr_metadata.pt')

# # # # ------------  dataset object ------------
# # # dataset = ECtHRDataset(root_dir='.', download=False, group_by_fields=['defendant_state', 'applicant_age', 'applicant_gender'])
# # # subset  = dataset.get_subset('test')       # or 'val', whichever matches your CSV

# # # # ------------  run evaluation ------------
# # # results, results_str = subset.eval(y_pred, y_true, metadata)
# # # print(results_str)

# # # # optional: save
# # # with open('ecthr_eval_summary.txt', 'w') as f:
# # #     f.write(results_str)

# # #!/usr/bin/env python3
# # """
# # Compute overall macro-F1 and per-group worst-/average F1
# # for ECtHR predictions already saved as *.pt tensors.

# # Inputs  (all produced by prep_ecthr_tensors_constants.py):
# #     <prefix>_y_true.pt      shape [N, 11]   multi-hot ground truth
# #     <prefix>_y_pred.pt      shape [N, 11]   multi-hot predictions
# #     <prefix>_metadata.pt    shape [N, 3]    [def_state, age, gender]

# # Outputs:
# #   • Pretty print of overall F1, worst-group F1, and per-group scores
# #   • `results.json` (same dict, ready for further use)
# # """

# # import argparse, json, itertools, numpy as np, torch
# # from sklearn.metrics import f1_score

# # # 1. ---------- CLI -------------------------------------------------
# # p = argparse.ArgumentParser()
# # p.add_argument("--prefix", required=True, help="Path prefix of the *.pt files")
# # args = p.parse_args()

# # y_true   = torch.load(f"{args.prefix}_y_true.pt").numpy()
# # y_pred   = torch.load(f"{args.prefix}_y_pred.pt").numpy()
# # metadata = torch.load(f"{args.prefix}_metadata.pt").numpy()   # int32 [N,3]

# # assert y_true.shape == y_pred.shape
# # N = y_true.shape[0]

# # # 2. ---------- overall macro-F1 ------------------------------------
# # overall_f1 = f1_score(y_true, y_pred, average="macro")

# # # 3. ---------- groupwise -------------------------------------------
# # # each row’s group key is a 3-tuple (def_state, age, gender)
# # group_keys, group_scores = [], {}
# # for i in range(N):
# #     group_keys.append(tuple(metadata[i]))

# # unique_keys = sorted(set(group_keys))
# # for key in unique_keys:
# #     idx = [j for j,k in enumerate(group_keys) if k == key]
# #     if idx:          # should always be true
# #         g_f1 = f1_score(y_true[idx], y_pred[idx], average="macro")
# #         group_scores[key] = g_f1

# # # 4. ---------- worst-group & average -------------------------------
# # worst_f1  = min(group_scores.values())
# # avg_f1    = float(np.mean(list(group_scores.values())))

# # # 5. ---------- present ---------------------------------------------
# # print("\n===== ECtHR evaluation (offline) =====")
# # print(f"Overall macro-F1 : {overall_f1:6.4f}")
# # print(f"Avg-group F1     : {avg_f1:6.4f}")
# # print(f"Worst-group F1   : {worst_f1:6.4f}")
# # print("\nPer-group F1 (def_state, age, gender) → score")
# # for k,v in group_scores.items():
# #     print(f"  {k} : {v:6.4f}")

# # # 6. ---------- dump to JSON ----------------------------------------
# # results = {
# #     "overall_macro_F1": overall_f1,
# #     "avg_group_F1":     avg_f1,
# #     "worst_group_F1":   worst_f1,
# #     "per_group":        {str(k): v for k,v in group_scores.items()}
# # }
# # with open(f"{args.prefix}_results.json", "w") as f:
# #     json.dump(results, f, indent=2)
# # print(f"\n✓ wrote {args.prefix}_results.json")


# #!/usr/bin/env python3

# import argparse, json, numpy as np, torch
# from sklearn.metrics import f1_score
# from collections import defaultdict
# from scipy.stats import entropy

# # ---------- CLI ----------
# p = argparse.ArgumentParser()
# p.add_argument("--prefix", required=True, help="Path prefix of the *.pt files")
# args = p.parse_args()

# # ---------- Load data ----------
# y_true   = torch.load(f"{args.prefix}_y_true.pt").numpy()
# y_pred   = torch.load(f"{args.prefix}_y_pred.pt").numpy()
# metadata = torch.load(f"{args.prefix}_metadata.pt").numpy()  # [N, 3]

# assert y_true.shape == y_pred.shape
# N = y_true.shape[0]

# # ---------- Overall F1 ----------
# overall_f1 = f1_score(y_true, y_pred, average="macro")

# # ---------- Full group evaluation (3-tuple) ----------
# group_keys = [tuple(row) for row in metadata]
# group_scores = {}
# for key in sorted(set(group_keys)):
#     idx = [i for i, k in enumerate(group_keys) if k == key]
#     if idx:
#         g_f1 = f1_score(y_true[idx], y_pred[idx], average="macro")
#         group_scores[key] = g_f1

# worst_f1 = min(group_scores.values())
# avg_f1   = float(np.mean(list(group_scores.values())))

# # ---------- Group-by-axis evaluations (drop 'n/a') ----------
# def axis_eval(axis_idx, axis_name, exclude_val):
#     axis_vals = metadata[:, axis_idx]
#     valid = axis_vals != exclude_val
#     valid_y_true   = y_true[valid]
#     valid_y_pred   = y_pred[valid]
#     valid_axis_val = axis_vals[valid]

#     scores = {}
#     for val in sorted(set(valid_axis_val)):
#         idx = valid_axis_val == val
#         scores[int(val)] = f1_score(valid_y_true[idx], valid_y_pred[idx], average="macro")

#     probs = np.array(list(scores.values()))
#     probs /= probs.sum()
#     uniform = np.ones_like(probs) / len(probs)
#     ldkl = entropy(probs, uniform)
#     wci  = 1 - min(scores.values())
#     return scores, ldkl, wci

# age_scores, age_ldkl, age_wci       = axis_eval(1, "age", exclude_val=3)     # "n/a"
# gender_scores, gender_ldkl, gender_wci = axis_eval(2, "gender", exclude_val=2)

# # ---------- Output ----------
# print("\n===== ECtHR Group Fairness Evaluation =====")
# print(f"Overall macro-F1 : {overall_f1:6.4f}")
# print(f"Avg-group F1     : {avg_f1:6.4f}")
# print(f"Worst-group F1   : {worst_f1:6.4f}")

# print("\n→ Per-group F1 (def_state, age, gender)")
# for k, v in group_scores.items():
#     print(f"  {k}: {v:.4f}")

# print("\n→ By age group (excluding 'n/a')")
# for k, v in age_scores.items():
#     print(f"  Age {k}: {v:.4f}")
# print(f"  LD_KL(age): {age_ldkl:.4f} | WCI(age): {age_wci:.4f}")

# print("\n→ By gender group (excluding 'n/a')")
# for k, v in gender_scores.items():
#     print(f"  Gender {k}: {v:.4f}")
# print(f"  LD_KL(gender): {gender_ldkl:.4f} | WCI(gender): {gender_wci:.4f}")

# # ---------- Save results ----------
# results = {
#     "overall_macro_F1": overall_f1,
#     "avg_group_F1": avg_f1,
#     "worst_group_F1": worst_f1,
#     "per_group": {str(k): v for k, v in group_scores.items()},
#     "age_scores": {str(k): v for k, v in age_scores.items()},
#     "age_LDKL": age_ldkl,
#     "age_WCI": age_wci,
#     "gender_scores": {str(k): v for k, v in gender_scores.items()},
#     "gender_LDKL": gender_ldkl,
#     "gender_WCI": gender_wci
# }

# with open(f"{args.prefix}_results.json", "w") as f:
#     json.dump(results, f, indent=2)
# print(f"\n✓ Saved to {args.prefix}_results.json")

#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
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

