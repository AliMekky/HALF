import os
import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
OUTPUT_DIR = 'fairlex_samples'
TOTAL_SAMPLE_SIZE = 1000

def sample_by_column(dataset_dict: DatasetDict, sample_size: int, stratify_col: str):
    """Sample from combined splits using stratification by a specified column."""
    all_dfs = [dataset_dict[split].to_pandas() for split in dataset_dict]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=[stratify_col])
    combined_df = combined_df[combined_df[stratify_col].apply(lambda x: isinstance(x, (int, float)))]
    combined_df[stratify_col] = combined_df[stratify_col].astype(int)
    (sampled_df, _) = train_test_split(combined_df, train_size=sample_size, stratify=combined_df[stratify_col], random_state=42)
    return sampled_df

def process_ecthr():
    dataset = load_dataset('coastalcph/fairlex', 'ecthr')
    sampled = sample_by_column(dataset, TOTAL_SAMPLE_SIZE, stratify_col='applicant_age')
    sampled.to_json(os.path.join(OUTPUT_DIR, 'ecthr_sampled.jsonl'), orient='records', lines=True)
    print('✅ Saved: ecthr_sampled.jsonl')




def main():
    process_ecthr()
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
