"""
prep_ecthr_tensors_constants.py

Convert:
  1. Ground-truth CSV with columns
       text, labels, applicant_age, applicant_gender, defendant_state
     where `labels` is a Python list of ints 0-9
       (empty list → no violation),
  2. ChatGPT batch JSONL (OpenAI Batch API format)

into three torch tensors:
  <prefix>_y_true.pt, <prefix>_y_pred.pt, <prefix>_metadata.pt
ready for ECtHRDataset.eval(), and counts hallucinated article-codes.
"""

import json
import ast
import argparse
import pandas as pd
import torch
from collections import Counter

# -------------------------------------------------------------------
# === 1.  CONSTANTS  =================================================
EAST_EUROPEAN_COUNTRIES = {
    'RUSSIA','TURKEY','UKRAINE','POLAND','BULGARIA','CROATIA','HUNGARY',
    'ROMANIA','SLOVAKIA','MOLDOVA','SLOVENIA','LITHUANIA','SERBIA',
    'AZERBAIJAN','CZECH REPUBLIC','GEORGIA','ESTONIA','BOSNIA & HERZEGOVINA',
    'NORTH MACEDONIA','FORMER YUGOSLAV MACEDONIA','ARMENIA','LATVIA','MONTENEGRO'
}

ECHR_ARTICLES = {
    "2":   "Right to life",
    "3":   "Prohibition of torture",
    "5":   "Right to liberty and security",
    "6":   "Right to a fair trial",
    "8":   "Right to respect for private and family life",
    "9":   "Freedom of thought, conscience and religion",
    "10":  "Freedom of expression",
    "11":  "Freedom of assembly and association",
    "14":  "Prohibition of discrimination",
    "P1-1":"Protection of property",
    "NV":  "No Violation",
}

# Ground-truth index order (0-9) matches the first 10 articles below
ARTICLE_LIST = ["2","3","5","6","8","9","10","11","14","P1-1","NV"]
N_CLASSES = len(ARTICLE_LIST)                         # 11

ID2IDX = {art: idx for idx, art in enumerate(ARTICLE_LIST)}

GENDERS    = {'n/a': 0, 'male': 1, 'female': 2}
AGE_GROUPS = {'n/a': 0, '<=35': 1, '<=65': 2, '>65': 3}
# -------------------------------------------------------------------

# Counter for hallucinated article-codes
unknown_counter = Counter()

# ---------- helpers ------------------------------------------------
def encode_gender(g):     return GENDERS.get(str(g).lower(), 0)
def encode_age(a):        return AGE_GROUPS.get(str(a), 0)
def encode_def_state(c):  # 0 = Rest of Europe, 1 = East-Europe group
    if not isinstance(c, str): return 0
    return 1 if c.strip().upper() in EAST_EUROPEAN_COUNTRIES else 0

# ---------- ground-truth vector -----------------------------------
def gt_indices_to_vec(index_list, row_idx):
    """
    index_list : list[int]  (0-9)   – empty list ⇒ NV
    Unknown indices are skipped and reported.
    """
    vec = torch.zeros(N_CLASSES, dtype=torch.float32)

    if not index_list:               # No violation
        vec[ID2IDX["NV"]] = 1.0
        return vec

    unknown = []
    for i in index_list:
        if 0 <= i < 10:
            vec[i] = 1.0
        else:
            unknown.append(i)

    if unknown:
        print(f"[GT row {row_idx}] skipped unknown indices: {unknown}")

    if vec.sum() == 0:               # all skipped → treat as NV
        vec[ID2IDX["NV"]] = 1.0
    return vec

# ---------- prediction vector -------------------------------------
def pred_to_vec(pred_field: str, row_idx: int):
    """
    ChatGPT `content` string  →  many-hot vector.
      • Accepts formats:  '8' | '5,8' | '[6]' | 'P1-1' | 'None' | 'NV' …
      • Unknown IDs are counted as hallucinations.
    """
    if pred_field is None or str(pred_field).strip().upper() in {"NONE", "NV"}:
        return torch.eye(N_CLASSES)[ID2IDX["NV"]]

    cleaned = str(pred_field).replace('[', '').replace(']', '')
    raw_ids = [tok.strip() for tok in cleaned.split(',') if tok.strip()]

    valid_ids, unknown = [], []
    for tok in raw_ids:
        candidate = tok.upper()
        if candidate.isdigit():
            candidate = candidate.lstrip("0") or "0"
        if candidate in ECHR_ARTICLES:
            valid_ids.append(candidate)
        else:
            unknown.append(tok)

    if unknown:
        # accumulate hallucinated tokens
        unknown_counter.update(unknown)

    if not valid_ids:
        return torch.eye(N_CLASSES)[ID2IDX["NV"]]

    vec = torch.zeros(N_CLASSES, dtype=torch.float32)
    for art in valid_ids:
        vec[ID2IDX[art]] = 1.0
    return vec

# ---------- main ---------------------------------------------------
def main(csv_path, jsonl_path, prefix):
    # 1. Load CSV ---------------------------------------------------
    df = pd.read_csv(csv_path)

    # 2. Build y_true ----------------------------------------------
    y_true = torch.stack([
        gt_indices_to_vec(ast.literal_eval(label_str), idx)
        for idx, label_str in enumerate(df['labels'])
    ])

    # 3. Build metadata --------------------------------------------
    metadata = torch.tensor([
        [
            row['defendant_state'],
            row['applicant_age'],
            row['applicant_gender']
        ]
        for _, row in df.iterrows()
    ], dtype=torch.long)

    # 4. Build y_pred ----------------------------------------------
    preds = []
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            if prefix == "deepseek":
                if "response" not in obj:
                    print(f"[PRED row {idx}] No response field found, skipping.")
                    continue
                content = obj['response']
            elif prefix == "claude":
                content = obj['result']["message"]["content"][0]["text"]
            else:
                content = obj['response']['body']['choices'][0]['message']['content']
            preds.append(pred_to_vec(content, idx))
    y_pred = torch.stack(preds)

    # 5. Report hallucinations --------------------------------------
    total_rows = len(preds)
    rows_with_halluc = len({i for i, c in enumerate(preds) if any(tok for tok in unknown_counter)})
    total_halluc_ids = sum(unknown_counter.values())
    print(f"\n✓ Hallucinated article-codes in {rows_with_halluc}/{total_rows} rows "
          f"({total_halluc_ids} total instances).")
    print("  Top hallucinated tokens:", unknown_counter.most_common(10))

    # 6. Sanity-check ----------------------------------------------
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
    )

    # 7. Save tensors ----------------------------------------------
    torch.save(y_pred,    f"{prefix}_y_pred.pt")
    torch.save(y_true,    f"{prefix}_y_true.pt")
    torch.save(metadata,  f"{prefix}_metadata.pt")
    print(f"✓ Saved tensors: {prefix}_y_pred.pt, {prefix}_y_true.pt, {prefix}_metadata.pt")

# ---------- CLI ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True, help="Ground-truth CSV path")
    parser.add_argument("--jsonl", required=True, help="ChatGPT batch JSONL path")
    parser.add_argument("--prefix", default="ecthr_eval", help="Output file prefix")
    args = parser.parse_args()
    main(args.csv, args.jsonl, args.prefix)
