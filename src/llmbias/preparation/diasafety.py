from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    dataset = load_dataset('thu-coai/diasafety')
    full_dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])
    biased_opinion_data = full_dataset.filter(lambda x: x['category'] == 'Biased Opinion')
    df = biased_opinion_data.to_pandas()
    print(df['label'].value_counts())
    (sampled_df, _) = train_test_split(df, train_size=1000, stratify=df['label'], random_state=42)
    print(sampled_df['label'].value_counts())
    sampled_df.to_csv('diasafety_biased_opinion_sampled.csv', index=False)
if __name__ == '__main__':
    main()
