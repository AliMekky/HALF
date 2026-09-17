import argparse
import copy
import json
import pandas as pd
from llmbias.prompts import SYSTEM_PROMPT_MENTAL_HEALTH, SYSTEM_PROMPT_MEDICAL, DATASETS, COMBINATIONS, MED_BULLETS_COMBINATIONS
from llmbias.compat import GENDERS, swap_gender, add_ethnicity_to_gender, functions
import re


def main():
    parser = argparse.ArgumentParser(description="Generate batch JSONL for LLM prompts.")
    parser.add_argument('--dataset', required=True, help="Name of the dataset")
    parser.add_argument('--model', default='gpt-4o', help="LLM model name (default: 'gpt')")
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    domain = dataset.split('/')[0]

    sample_path = f"../{dataset}.csv"
    dataset = dataset.split('/')[-1]
    if dataset not in functions:
        raise ValueError(f"Dataset {dataset} is outside the paper scope. Choose from {sorted(functions)}")
    df = pd.read_csv(sample_path)

    base_template_dic = {
        "custom_id": None,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": None,
                },
                {
                    "role": "user",
                    "content": None,
                }
            ]
        }
    }

    # Add temperature only if model is not GPT-4o
    if "o4" not in model.lower():
        base_template_dic["body"]["temperature"] = 0.6

    batch_result = []
    if dataset == "medbullets" or dataset == "djinni" or dataset == "CAMS" or dataset == "SAD" or dataset == "movielens":
        batch_result, neutral_batch_results = functions[dataset](df, model, dataset, base_template_dic)
        with open(f"./{domain}/batch_files/{model}_{dataset}_neutral.jsonl", 'w') as f:
            for entry in neutral_batch_results:
                json.dump(entry, f)
                f.write('\n')
    elif dataset in DATASETS:
        batch_result = functions[dataset](df, model, dataset, base_template_dic)
    else:
        raise ValueError(f"Dataset {dataset} is not supported. Please choose from {DATASETS.keys()}.")
    output_path = f"./{domain}/batch_files/{model}_{dataset}.jsonl"
    with open(output_path, 'w') as f:
        for entry in batch_result:
            json.dump(entry, f)
            f.write('\n')
            


    print(f"Batch file written to: {output_path}")

if __name__ == "__main__":
    main()


