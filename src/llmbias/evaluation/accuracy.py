import pandas as pd
import argparse

def compute_accuracy(gt_path, pred_path, group_cols=None):
    # Load CSV files
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    # Merge on index, gender, and ethnicity if present
    merge_cols = ["index"]
    for col in ["gender", "ethnicity"]:
        if col in gt_df.columns and col in pred_df.columns:
            merge_cols.append(col)

    merged = pd.merge(gt_df, pred_df, on=merge_cols, suffixes=('_gt', '_pred'))

    # Compute match
    merged["correct"] = merged["answer_gt"] == merged["answer_pred"]

    # Overall accuracy
    overall_acc = merged["correct"].mean()
    print(f"✅ Overall Accuracy: {overall_acc:.3f}")

    # Optional: grouped accuracy
    if group_cols:
        print("\n📊 Grouped Accuracy:")
        group_acc = merged.groupby(group_cols)["correct"].mean().reset_index()
        print(group_acc.to_string(index=False))

    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_csv", required=True, help="Path to ground truth CSV")
    parser.add_argument("--pred_csv", required=True, help="Path to model output CSV")
    parser.add_argument("--group_by", nargs="*", default=[], help="Group by columns, e.g., gender ethnicity")
    args = parser.parse_args()

    compute_accuracy(args.gt_csv, args.pred_csv, args.group_by)
