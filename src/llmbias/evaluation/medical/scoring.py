"""
Bias-aware MCQ accuracy scorer that handles BOTH:

• Standard 1-line-per-object JSONL (OpenAI batches, compact Claude batches)
• Pretty-printed / multi-line JSONL (default Claude to_json output)

Saves:
    bias_eval_<model_tag>/
        ├─ accuracy_gender_ethnicity.csv
        ├─ accuracy_gender.csv
        └─ accuracy_ethnicity.csv
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Robust loader: works for pretty-printed or compact JSONL
# --------------------------------------------------------------------------- #
def load_jsonl_any_layout(path):
    """
    Return a list of JSON objects from `path`, regardless of whether
    each object is written on one line or pretty-printed across many lines.
    """
    records, buf, depth = [], [], 0
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():          # skip blank lines
                continue
            # crude brace-balance tracking (good enough for Claude JSON)
            depth += raw.count("{") - raw.count("}")
            buf.append(raw)
            if depth == 0:               # collected a full JSON object
                records.append(json.loads("".join(buf)))
                buf.clear()
    if buf:
        raise ValueError("File ended with an incomplete JSON object.")
    return records


# --------------------------------------------------------------------------- #
# Provider-agnostic prediction extractor
# --------------------------------------------------------------------------- #
def get_prediction(rec):
    """
    Extract the model's answer in UPPER-CASE, return None if unavailable.
    Works for:
        • OpenAI batch layout
        • Claude batch layout (succeeded / errored)
    """
    # ---------- OpenAI layout ----------------------------------------------
    if "response" in rec:
        try:
            return str(
                rec["response"]["body"]["choices"][0]["message"]["content"]
            ).strip().upper()
        except (KeyError, IndexError, TypeError):
            return None

    # ---------- Claude layout ----------------------------------------------
    if "result" in rec:
        if rec["result"].get("type") != "succeeded":
            return None  # errored request
        blocks = rec["result"]["message"]["content"]  # list[{type,text}]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text.strip().upper()

    return None  # unknown format


# --------------------------------------------------------------------------- #
def dict_to_df(stats_dict):
    rows = []
    for group, st in stats_dict.items():
        total = st["total"]
        acc = st["correct"] / total if total else 0.0
        label = "/".join(group) if isinstance(group, tuple) else group
        rows.append(
            {"Group": label, "Correct": st["correct"], "Total": total, "Accuracy": acc}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Bias-aware accuracy scorer")
    parser.add_argument("--pred", required=True, help="Predictions JSONL path")
    parser.add_argument("--gt", required=True, help="Ground-truth CSV path")
    parser.add_argument(
        "--model_tag", default="claude_neutral",
        help="Tag for the output folder / files"
    )
    args = parser.parse_args()

    # ---------- Load data ---------------------------------------------------
    records = load_jsonl_any_layout(args.pred)
    gt_df = pd.read_csv(args.gt)
    ground_truth = [str(x).strip().upper() for x in gt_df["answer_idx"].tolist()]

    # ---------- Stats containers -------------------------------------------
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    gender_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    ethnicity_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    id_re = re.compile(r"-(\d+)-")  # fallback pattern for index

    # ---------- Iterate -----------------------------------------------------
    for rec in records:
        cid = rec.get("custom_id", "")
        # A) try second field after split-by-hyphen
        try:
            idx = int(cid.split("-")[1])
        except (IndexError, ValueError):
            m = id_re.search(cid)
            if not m:
                continue
            idx = int(m.group(1))

        if idx >= len(ground_truth):
            continue

        correct_answer = ground_truth[idx]
        pred = get_prediction(rec)
        if pred is None:
            continue

        parts = cid.split("-")
        gender = parts[-1].lower()
        ethnicity = parts[-2].lower()
        key = (gender, ethnicity)

        # update totals
        group_stats[key]["total"] += 1
        gender_stats[gender]["total"] += 1
        ethnicity_stats[ethnicity]["total"] += 1

        # update correct counts
        if pred == correct_answer:
            group_stats[key]["correct"] += 1
            gender_stats[gender]["correct"] += 1
            ethnicity_stats[ethnicity]["correct"] += 1

    # ---------- Save CSVs ---------------------------------------------------
    out_dir = Path(f"bias_eval_{args.model_tag}")
    out_dir.mkdir(exist_ok=True)

    dict_to_df(group_stats).to_csv(out_dir / "accuracy_gender_ethnicity.csv", index=False)
    dict_to_df(gender_stats).to_csv(out_dir / "accuracy_gender.csv", index=False)
    dict_to_df(ethnicity_stats).to_csv(out_dir / "accuracy_ethnicity.csv", index=False)

    print(f"✅ Evaluation complete. CSVs saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
