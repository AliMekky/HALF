from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the DiaSafety dataset
dataset = load_dataset("thu-coai/diasafety")

# Concatenate all splits
full_dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])

# Filter only "Biased Opinion" examples
biased_opinion_data = full_dataset.filter(lambda x: x['category'] == 'Biased Opinion')

# Convert to pandas for easier stratified sampling
df = biased_opinion_data.to_pandas()

print(df['label'].value_counts())

# Perform stratified sampling of 1000 examples based on the 'label' column
sampled_df, _ = train_test_split(
    df,
    train_size=1000,
    stratify=df['label'],
    random_state=42
)

print(sampled_df['label'].value_counts())

sampled_df.to_csv("diasafety_biased_opinion_sampled.csv", index=False)
