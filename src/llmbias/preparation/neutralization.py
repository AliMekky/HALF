from llmbias.paths import workspace_path
import copy
import pandas as pd
import json
from tqdm import tqdm


# ================================
# ✅ System Prompt for Neutralization
# ================================

SYSTEM_PROMPT_NEUTRALIZATION = """
You are an expert assistant specialized in anonymizing and neutralizing texts while preserving their original meaning.

Your task is to rewrite any given text to remove or neutralize sensitive demographic information, including:
- Countries, nationalities, and ethnicities (replace with a neutral phrase like "my country" or "a person from my country").
- Age (remove any mention of specific ages or phrases like "22-year-old" or "teenager").
- Gender-specific words (like he/she, male/female, man/woman, girl/boy) should be converted into neutral forms (they, person, individual, partner, etc.).
- Relationship-specific gendered terms (boyfriend/girlfriend, husband/wife) should be replaced with neutral terms like "partner" or "significant other."
- Religious affiliations (Christian, Muslim, Jewish, etc.) should be removed. If relevant, replace with "religious person" only if needed for context.
- Remove any pronoun gender bias (he/she → they, him/her → them).
- Neutralize any other potentially sensitive information (e.g., disabilities, socioeconomic status).

Output only the rewritten, neutralized text. Do not explain or add anything else.
"""

# ================================
# ✅ Base Template Example
# ================================

BASE_TEMPLATE = {
    "custom_id": "",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4.1-mini-2025-04-14",  # Change to your model
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""}
        ],
        "temperature": 0.5
    }
}

# ================================
# ✅ Function to Build Batch File
# ================================

def create_neutralization_batch(df, model, output_path):
    batch_result = []

    for i in tqdm(range(len(df))):
        text = df.iloc[i]['text']

        template = copy.deepcopy(BASE_TEMPLATE)
        template['custom_id'] = f"request-{i}-{model}-neutralization"
        template['body']['model'] = model
        template['body']['messages'][0]['content'] = SYSTEM_PROMPT_NEUTRALIZATION

        template['body']['messages'][1]['content'] = (
            f"Original Text: {text}\n\n"
            "Please rewrite this text following the instructions to neutralize all sensitive demographic information and PLEASE DO NOT ADD ANY MORE INFORMATION ONLY REPLY WITH THE NEUTRALIZED TEXT."
        )

        batch_result.append(template)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in batch_result:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ Batch file saved to {output_path} with {len(batch_result)} requests.")


# ================================
# ✅ Example Run
# ================================

if __name__ == "__main__":
    # 🔧 Load your dataset (CSV or TSV with a column 'text')
    input_file = str(workspace_path('repo/LLMBias/mental_health_data/dreaddit.csv'))   # Change to your file
    output_file = "neutralization_batch_dreaddit.jsonl"
    model_name = "gpt-4.1-mini-2025-04-14"           # Change to your model

    # 🔥 Load DataFrame
    if input_file.endswith('.tsv'):
        df = pd.read_csv(input_file, sep='\t')
    else:
        df = pd.read_csv(input_file)

    assert 'text' in df.columns, "❌ The input file must have a 'text' column."

    # 🚀 Run batch creation
    create_neutralization_batch(df, model=model_name, output_path=output_file)
