import os
import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "fairlex_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_SAMPLE_SIZE = 1000


def sample_by_column(dataset_dict: DatasetDict, sample_size: int, stratify_col: str):
    """Sample from combined splits using stratification by a specified column."""
    all_dfs = [dataset_dict[split].to_pandas() for split in dataset_dict]
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Drop rows with missing or invalid values
    combined_df = combined_df.dropna(subset=[stratify_col])
    combined_df = combined_df[combined_df[stratify_col].apply(lambda x: isinstance(x, (int, float)))]

    # Cast to int if needed
    combined_df[stratify_col] = combined_df[stratify_col].astype(int)

    sampled_df, _ = train_test_split(
        combined_df,
        train_size=sample_size,
        stratify=combined_df[stratify_col],
        random_state=42
    )
    return sampled_df


# === Dataset-specific functions ===

def process_ecthr():
    dataset = load_dataset("coastalcph/fairlex", "ecthr")
    sampled = sample_by_column(dataset, TOTAL_SAMPLE_SIZE, stratify_col="applicant_age")
    sampled.to_json(os.path.join(OUTPUT_DIR, "ecthr_sampled.jsonl"), orient="records", lines=True)
    print("✅ Saved: ecthr_sampled.jsonl")


def process_scotus():
    dataset = load_dataset("coastalcph/fairlex", "scotus")
    sampled = sample_by_column(dataset, TOTAL_SAMPLE_SIZE, stratify_col="decision_direction")
    sampled.to_json(os.path.join(OUTPUT_DIR, "scotus_sampled.jsonl"), orient="records", lines=True)
    print("✅ Saved: scotus_sampled.jsonl")


def process_fscs():
    dataset = load_dataset("coastalcph/fairlex", "fscs")
    sampled = sample_by_column(dataset, TOTAL_SAMPLE_SIZE, stratify_col="decision_language")
    sampled.to_json(os.path.join(OUTPUT_DIR, "fscs_sampled.jsonl"), orient="records", lines=True)
    print("✅ Saved: fscs_sampled.jsonl")


def process_cail():
    dataset = load_dataset("coastalcph/fairlex", "cail")
    sampled = sample_by_column(dataset, TOTAL_SAMPLE_SIZE, stratify_col="defendant_gender")
    sampled.to_json(os.path.join(OUTPUT_DIR, "cail_sampled.jsonl"), orient="records", lines=True)
    print("✅ Saved: cail_sampled.jsonl")


def main():
    process_ecthr()
    process_scotus()
    process_fscs()
    process_cail()


if __name__ == "__main__":
    main()
