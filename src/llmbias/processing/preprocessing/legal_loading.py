# load_dataset.py

import os
import pandas as pd

DATA_DIR = "fairlex_samples"
DATASETS = ["ecthr"]


def load_sampled_dataset(dataset_name):
    file_path = os.path.join(DATA_DIR, f"{dataset_name}_sampled.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    df = pd.read_json(file_path, lines=True)
    print(f"✅ Loaded {dataset_name}: {df.shape[0]} samples")
    return df


def load_all_datasets():
    data = {}
    for dataset_name in DATASETS:
        try:
            data[dataset_name] = load_sampled_dataset(dataset_name)
        except FileNotFoundError as e:
            print(e)
    return data


if __name__ == "__main__":
    datasets = load_all_datasets()
