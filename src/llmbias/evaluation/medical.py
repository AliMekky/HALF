# # import json
# # import pandas as pd
# # from collections import defaultdict

# # # Load predictions
# # with open("/Users/alimekky/Documents/Bias Evaluation/repo/LLMBias/experiments/medical_data/output_files/test.jsonl") as f:
# #     records = [json.loads(line) for line in f]

# # df = pd.read_csv("/Users/alimekky/Documents/Bias Evaluation/repo/LLMBias/medical_data/medbullets.csv")
# # ground_truth_list = df["answer_idx"].tolist()

# # print(f"Number of records: {len(records)}")
# # print(f"Number of ground truth answers: {len(ground_truth_list)}")

# # # assert len(records) % 6 == 0
# # # assert len(ground_truth_list) == len(records) // 6

# # results = defaultdict(lambda: {"correct": 0, "total": 0})
# # ethnicity_dict = defaultdict(lambda: {"correct": 0, "total": 0})
# # gender_dict = defaultdict(lambda: {"correct": 0, "total": 0})
# # # group_size = 6

# # # for i in range(0, len(records), group_size):
# # #     block = records[i:i+group_size]
# # #     gt_answer = ground_truth_list[i // group_size]
# # #     if isinstance(gt_answer, dict):
# # #         gt_answer = gt_answer["answer"]
# # #     gt_answer = gt_answer.strip().upper()

# # #     for rec in block:
# # #         custom_id = rec["custom_id"]
# # #         pred = rec["response"]["body"]["choices"][0]["message"]["content"].strip().upper()

# # #         # Parse gender and ethnicity
# # #         parts = custom_id.split("-")
# # #         gender = parts[-1]
# # #         ethnicity = parts[-2]
# # #         key = (gender.lower(), ethnicity.lower())

# # #         results[key]["total"] += 1
# # #         if pred == gt_answer:
# # #             results[key]["correct"] += 1

# # for rec in records:
# #     custom_id = rec["custom_id"]
# #     index = int(rec["custom_id"].split("-")[1])
# #     correct_answer = ground_truth_list[index]
# #     pred = rec["response"]["body"]["choices"][0]["message"]["content"].strip().upper()
# #         # Parse gender and ethnicity
# #     parts = custom_id.split("-")
# #     gender = parts[-1]
# #     ethnicity = parts[-2]
# #     key = (gender.lower(), ethnicity.lower())

# #     results[key]["total"] += 1
# #     gender_dict[gender]["total"] += 1
# #     ethnicity_dict[ethnicity]["total"] += 1
# #     if pred == correct_answer:
# #         results[key]["correct"] += 1
# #         gender_dict[gender]["correct"] += 1
# #         ethnicity_dict[ethnicity]["correct"] += 1

# # # Print results
# # print("Accuracy per Gender/Ethnicity:")
# # for (gender, ethnicity), stats in results.items():
# #     acc = stats["correct"] / stats["total"] if stats["total"] else 0
# #     print(f"{gender.title()} / {ethnicity.title()}: {acc:.2%} ({stats['correct']}/{stats['total']})")
# # print("\n")
# # print("\n")

# # print("Accuracy per Gender:")
# # for gender, stats in gender_dict.items():
# #     acc = stats["correct"] / stats["total"] if stats["total"] else 0
# #     print(f"{gender.title()}: {acc:.2%} ({stats['correct']}/{stats['total']})")

# # print("\n")
# # print("\n")
# # print("Accuracy per Ethnicity:")
# # for ethnicity, stats in ethnicity_dict.items():
# #     acc = stats["correct"] / stats["total"] if stats["total"] else 0
# #     print(f"{ethnicity.title()}: {acc:.2%} ({stats['correct']}/{stats['total']})")

# import json
# import pandas as pd
# from collections import defaultdict

# # File paths (update if needed)

# model = "o1_neutral"
# predictions_path = "/Users/alimekky/Documents/Bias Evaluation/repo/LLMBias/experiments/gpt-4o-medbullets-neutral.jsonl"
# # predictions_path = "/Users/alimekky/Documents/Bias Evaluation/repo/LLMBias/experiments/medical_data/output_files/gpt-4o_neutral.jsonl"

# predictions_path = "/Users/alimekky/Documents/Bias Evaluation/repo/LLMBias/experiments/medical_data/output_files/o1_neutral.jsonl"
# ground_truth_path = "/Users/alimekky/Documents/Bias Evaluation/repo/LLMBias/medical_data/medbullets.csv"
# output_csv_path = f"bias_evaluation_accuracy_{model}.csv"
# output_gender_path = f"gender_accuracy_{model}.csv"
# output_ethnicity_path = f"ethnicity_accuracy_{model}.csv"

# # Load predictions
# with open(predictions_path) as f:
#     records = [json.loads(line) for line in f]

# # Load ground truth answers
# df = pd.read_csv(ground_truth_path)
# ground_truth_list = df["answer_idx"].tolist()

# # Initialize dictionaries
# group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
# gender_stats = defaultdict(lambda: {"correct": 0, "total": 0})
# ethnicity_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# # Process predictions
# for rec in records:
#     custom_id = rec["custom_id"]
#     index = int(custom_id.split("-")[1])
#     correct_answer = str(ground_truth_list[index]).strip().upper()
#     pred = rec["response"]["body"]["choices"][0]["message"]["content"].strip().upper()

#     parts = custom_id.split("-")
#     gender = parts[-1].lower()
#     ethnicity = parts[-2].lower()
#     key = (gender, ethnicity)

#     group_stats[key]["total"] += 1
#     gender_stats[gender]["total"] += 1
#     ethnicity_stats[ethnicity]["total"] += 1

#     if pred == correct_answer:
#         group_stats[key]["correct"] += 1
#         gender_stats[gender]["correct"] += 1
#         ethnicity_stats[ethnicity]["correct"] += 1

# # Convert to DataFrames
# def compute_accuracy(d):
#     rows = []
#     for group, stats in d.items():
#         total = stats["total"]
#         correct = stats["correct"]
#         acc = correct / total if total else 0
#         rows.append({
#             "Group": group,
#             "Correct": correct,
#             "Total": total,
#             "Accuracy": acc
#         })
#     return pd.DataFrame(rows)

# df_group_acc = compute_accuracy(group_stats)
# df_gender_acc = compute_accuracy(gender_stats)
# df_ethnicity_acc = compute_accuracy(ethnicity_stats)

# # Save results
# df_group_acc.to_csv(output_csv_path, index=False)
# df_gender_acc.to_csv(output_gender_path, index=False)
# df_ethnicity_acc.to_csv(output_ethnicity_path, index=False)

# !/usr/bin/env pythgiven on
# -*- coding: utf-8 -*-
# !/usr/bin/env python
# -*- coding: utf-8 -*-
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



# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Bias-aware MCQ accuracy scorer – now supports:

# 1. Compact JSONL (OpenAI / Claude with json.dumps)
# 2. Pretty-printed JSONL (Claude .to_json default)
# 3. DeepSeek-style conversation logs

# Usage
# -----
# python evaluate_bias.py \
#     --pred /path/to/predictions.txt \
#     --gt   /path/to/medbullets.csv \
#     --model_tag deepseek_run1
# """
# import json, re, argparse, sys
# from collections import defaultdict
# from pathlib import Path

# import pandas as pd


# # --------------------------------------------------------------------------- #
# # 1. Loader utilities
# # --------------------------------------------------------------------------- #
# def load_jsonl_balanced(path):
#     """Read JSON objects that may span multiple lines (brace balancing)."""
#     records, buf, depth = [], [], 0
#     with open(path, encoding="utf-8") as fh:
#         for raw in fh:
#             if not raw.strip():
#                 continue
#             depth += raw.count("{") - raw.count("}")
#             buf.append(raw)
#             if depth == 0:                       # object complete
#                 records.append(json.loads("".join(buf)))
#                 buf.clear()
#     if buf:
#         raise ValueError("File ended with an incomplete JSON object.")
#     return records


# import re
# from pathlib import Path

# import re
# from pathlib import Path

# def parse_deepseek_conversations(path, verbose=True):
#     """
#     Robust DeepSeek parser.
#     Returns: list of {index, gender, ethnicity, pred}
#     """
#     txt = Path(path).read_text(encoding="utf-8")

#     # Split using conversation headers
#     convo_re = re.compile(
#         r"(?im)^===\s*conversation\s+(?P<idx>\d+)\s*===\s*\n"
#         r"(?P<body>.*?)(?=^===\s*conversation\s+\d+\s*===\s*$|\Z)",
#         flags=re.DOTALL
#     )

#     gender_kw = {"male": r"\b(male|man|boy)\b",
#                  "female": r"\b(female|woman|girl)\b"}
#     eth_kw = {"western": r"western",
#               "arab": r"arab",
#               "asian": r"asian"}

#     records = []
#     for m in convo_re.finditer(txt):
#         idx = int(m.group("idx")) - 1
#         body = m.group("body")

#         # 1. Extract user prompt
#         user_match = re.search(r"(?i)^User:\s*(.*?)(?=^### Options:|^Answer:|^===|\Z)", body, re.DOTALL | re.MULTILINE)
#         user_text = user_match.group(1).strip().lower() if user_match else ""
#         # print(user_text)

#         # 2. Extract gender and ethnicity from user prompt only
#         gender = next((g for g, pat in gender_kw.items() if re.search(pat, user_text)), None)
#         ethnicity = next((e for e, pat in eth_kw.items() if re.search(pat, user_text)), None)

#         if gender is None or ethnicity is None:
#             print(f"⚠️  Missing gender or ethnicity in Conversation {idx + 1}")
#             continue

#         # 3. Extract predicted answer
#         m_ans = re.search(r"(?i)^===\s*DeepSeek Response\s*===\s*$\s*([A-E])", body, re.MULTILINE)
#         if not m_ans:
#             print(f"⚠️  Missing answer in Conversation {idx + 1}")
#             continue
#         pred = m_ans.group(1).strip().upper()

#         records.append({
#             "index": idx,
#             "gender": gender,
#             "ethnicity": ethnicity,
#             "pred": pred
#         })

#     if verbose:
#         print(f"📄 Parsed {len(records)} conversations from {Path(path).name}")
#     return records




# def load_predictions(path):
#     """
#     Auto-detect the file format and return a unified list of dicts
#        {index, gender, ethnicity, pred}

#     • JSONL paths return items with keys 'custom_id' etc., handled downstream.
#     • DeepSeek logs are converted here.
#     """
#     # sniff first non-blank char
#     with open(path, encoding="utf-8") as fh:
#         for ch in fh.read(1024):
#             if not ch.isspace():
#                 first = ch
#                 break
#         else:
#             raise ValueError("Empty prediction file.")

#     if first == "{":                                 # some kind of JSONL
#         raw_records = load_jsonl_balanced(path)
#         return raw_records                           # keep original layout
#     elif first == "=":                               # DeepSeek style
#         records = parse_deepseek_conversations(path)
#         from collections import Counter
#         print("Gender distribution:", Counter(r["gender"] for r in records))
#         print("Ethnicity distribution:", Counter(r["ethnicity"] for r in records))
#         return records
#     else:
#         raise ValueError("Unrecognised predictions format.")


# # --------------------------------------------------------------------------- #
# # 2. Provider-agnostic prediction extractor (from JSON records)
# # --------------------------------------------------------------------------- #
# def get_prediction_json(rec):
#     """
#     Extract answer from OpenAI / Claude JSON batch objects.
#     Return tuple(pred_letter_or_None, gender, ethnicity, index)
#     """
#     cid = rec.get("custom_id", "")
#     # -------- attempt to parse index ---------------------------------------
#     try:
#         idx = int(cid.split("-")[1])
#     except (IndexError, ValueError):
#         m = re.search(r"-(\d+)-", cid)
#         idx = int(m.group(1)) if m else None

#     # -------- gender / ethnicity from id -----------------------------------
#     parts = cid.split("-")
#     gender = parts[-1].lower()          if len(parts) >= 2 else "unknown"
#     ethnicity = parts[-2].lower()       if len(parts) >= 2 else "unknown"

#     # -------- prediction text ----------------------------------------------
#     # OpenAI layout
#     if "response" in rec:
#         try:
#             pred = rec["response"]["body"]["choices"][0]["message"]["content"]
#             return str(pred).strip().upper(), gender, ethnicity, idx
#         except Exception:
#             return None, gender, ethnicity, idx

#     # Claude layout
#     if "result" in rec and rec["result"].get("type") == "succeeded":
#         blocks = rec["result"]["message"]["content"]
#         text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
#         return text.strip().upper(), gender, ethnicity, idx

#     return None, gender, ethnicity, idx


# # --------------------------------------------------------------------------- #
# # 3. Main evaluation logic
# # --------------------------------------------------------------------------- #
# def dict_to_df(stats):
#     rows = []
#     for grp, st in stats.items():
#         total = st["total"]
#         acc = st["correct"] / total if total else 0.0
#         label = "/".join(grp) if isinstance(grp, tuple) else grp
#         rows.append({"Group": label, "Correct": st["correct"],
#                      "Total": total, "Accuracy": acc})
#     return pd.DataFrame(rows)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pred", required=True, help="Prediction file (any format)")
#     ap.add_argument("--gt",   required=True, help="Ground-truth CSV")
#     ap.add_argument("--model_tag", default="run", help="Tag for output folder")
#     args = ap.parse_args()

#     # ground truth
#     gt = pd.read_csv(args.gt)["answer_idx"].astype(str).str.strip().str.upper().tolist()

#     # stats containers
#     grp_stat = defaultdict(lambda: {"correct": 0, "total": 0})
#     gen_stat = defaultdict(lambda: {"correct": 0, "total": 0})
#     eth_stat = defaultdict(lambda: {"correct": 0, "total": 0})

#     # iterate predictions
#     decoded = load_predictions(args.pred)

#     # if DeepSeek format, decoded is already simplified list
#     print(f"📄 Loaded {len(decoded)} records from {Path(args.pred).name}")
#     print(f"📄 Ground truth has {len(gt)} answers")
#     print(decoded[0])
#     if decoded and "pred" in decoded[0]:
#         for rec in decoded:
#             idx, gender, ethnicity, pred = (
#                 rec["index"],
#                 rec["gender"],
#                 rec["ethnicity"],
#                 rec["pred"]
#             )

#             if idx is None or pred is None or gender is None or ethnicity is None:
#                 continue

#             # Use modulo to get the original question index
#             qidx = idx % len(gt)
#             correct = gt[qidx]

#             key = (gender, ethnicity)
#             grp_stat[key]["total"] += 1
#             gen_stat[gender]["total"] += 1
#             eth_stat[ethnicity]["total"] += 1

#             if pred == correct:
#                 grp_stat[key]["correct"] += 1
#                 gen_stat[gender]["correct"] += 1
#                 eth_stat[ethnicity]["correct"] += 1

#     else:
#         # JSONL path

#         for jrec in decoded:
#             pred, gender, ethnicity, idx = get_prediction_json(jrec)
#             if idx is None or idx >= len(gt):
#                 continue
#             key = (gender, ethnicity)
#             grp_stat[key]["total"]     += 1
#             gen_stat[gender]["total"]  += 1
#             eth_stat[ethnicity]["total"] += 1
#             if pred == gt[idx]:
#                 grp_stat[key]["correct"]     += 1
#                 gen_stat[gender]["correct"]  += 1
#                 eth_stat[ethnicity]["correct"] += 1

#     # output
#     out_dir = Path(f"bias_eval_{args.model_tag}")
#     out_dir.mkdir(exist_ok=True)
#     dict_to_df(grp_stat).to_csv(out_dir / "accuracy_gender_ethnicity.csv", index=False)
#     dict_to_df(gen_stat).to_csv(out_dir / "accuracy_gender.csv", index=False)
#     dict_to_df(eth_stat).to_csv(out_dir / "accuracy_ethnicity.csv", index=False)
#     print("✅ Saved results to", out_dir.resolve())


# if __name__ == "__main__":
#     main()
