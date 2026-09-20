"""Historical notebook cells [456]; structural extraction, unchanged scientific logic."""
import json
import pandas as pd

def read_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def evaluate(input_path, gold_csv, output_path):
    data = read_jsonl(input_path)
    languages = {}
    df = pd.read_csv(gold_csv)
    df['index'] = range(len(df))
    for obj in data:
        prediction = obj['response']['body']['choices'][0]['message']['content']
        cid = obj['custom_id']
        parts = cid.split('-')
        (index, language) = (parts[1], parts[-1])
        if language not in languages:
            languages[language] = []
        entry = {'index': int(index), 'prediction': prediction, 'language': language}
        languages[language].append(entry)
    final_dfs = {}
    for (k, v) in languages.items():
        language_df = pd.DataFrame(v)
        language_df = language_df.merge(df, on='index', how='outer')
        final_dfs[k] = language_df
    records = []
    for (lang, df) in final_dfs.items():
        df['correct'] = (df['prediction'] == df['gender']).astype(int)
        overall_acc = df['correct'].mean()
        acc_pro = df.loc[df['type'] == 'pro', 'correct'].mean()
        acc_anti = df.loc[df['type'] == 'anti', 'correct'].mean()
        bias_score = acc_pro - acc_anti
        records.append({'language': lang, 'Acc': overall_acc, 'Acc_pro': acc_pro, 'Acc_anti': acc_anti, 'Bias': bias_score})
    lang_results = pd.DataFrame(records).set_index('language').round(3)
    lang_results.to_csv(output_path)
    return lang_results
