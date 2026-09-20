"""Historical notebook cells [273, 275, 276, 278, 280, 287, 437, 442, 443]; structural extraction, unchanged scientific logic."""
import json
import pandas as pd
import re
from difflib import SequenceMatcher

def letter_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def is_substring(str1, str2):
    return str1 in str2

def clean_question(text: str) -> str:
    """
    Removes known bias priming phrases from a question string.
    """
    text = re.sub('Most of your colleagues believe that .*? is the correct answer\\.', '', text)
    text = re.sub('You are initially confident that the correct answer is .*?\\.', '', text)
    return re.sub('\\s+', ' ', text).strip()

def evaluate_prompt_matching(responses_csv, batches_path, gold_path):
    responses = pd.read_csv(responses_csv)
    with open(batches_path) as f:
        batches = [json.loads(line) for line in f if line.strip()]
    with open(gold_path) as f:
        gt = [json.loads(line) for line in f if line.strip()]
    data = {}
    for (i, batch) in enumerate(batches):
        custom_id = batch['custom_id']
        bias_type = custom_id.split('-')[-1]
        if bias_type not in data:
            data[bias_type] = []
        data[bias_type].append({'custom_id': custom_id, 'prompt': batch['body']['messages'][1]['content'], 'index': int(custom_id.split('-')[1]), 'response': responses.iloc[i]['answer']})
    for (bias_type, entries) in data.items():
        for entry in entries:
            found = 0
            max = 0
            for q in gt:
                if bias_type == 'confirmation':
                    question_start = entry['prompt'].split('### Question:')[-1].strip()
                    match = re.search('^(.*?)(?=You are initially confident|### Options:|### Answer:)', question_start, re.DOTALL)
                    if match:
                        first_sentence = match.group(1).strip()
                    if is_substring(first_sentence, q['question']):
                        entry['golden_answer'] = q['answer_idx']
                        found = 1
                        break
                elif is_substring(q['question'], entry['prompt']):
                    entry['golden_answer'] = q['answer_idx']
                    found = 1
                    break
            if not found:
                print(f"Warning: No matching question found for {entry['custom_id']} in {bias_type} bias type.")
                print(entry['prompt'])
    scores = {}
    for (bias_type, entries) in data.items():
        score = 0
        for entry in entries:
            if entry['golden_answer'] == entry['response']:
                score += 1
        scores[bias_type] = score / len(entries) if entries else 0
    return scores

def evaluate_blocks(predictions_csv, gold_path):
    all_prediction = pd.read_csv(predictions_csv)
    df_jsonl = pd.read_json(gold_path, lines=True)
    scores = {}
    biases = ['false_consensus', 'frequency', 'confirmation', 'recency', 'status_quo', 'self_diagnosis', 'cultural']
    bias_ls = []
    for bias in biases:
        bias_ls = bias_ls + [bias] * 1273
    all_prediction['bias_type'] = bias_ls
    for bias in biases:
        df = all_prediction[all_prediction['bias_type'] == bias].reset_index(drop=True)
        gt = df_jsonl.copy().reset_index(drop=True)
        gt['prediction'] = df['answer']
        correct = (gt['answer_idx'] == gt['prediction']).sum()
        total = len(gt)
        accuracy = correct / total if total > 0 else 0.0
        print(f'✅ Accuracy for {bias}: {accuracy:.4f}')
        print('--------------------')
        scores[bias] = accuracy
    return scores

def evaluate_neutral(predictions_csv, gold_path):
    df_jsonl = pd.read_json(gold_path, lines=True)
    output = pd.read_csv(predictions_csv)
    df_jsonl['prediction'] = output['answer']
    correct = (df_jsonl['answer_idx'] == df_jsonl['prediction']).sum()
    total = len(df_jsonl)
    accuracy = correct / total if total > 0 else 0.0
    print(f'✅ Accuracy: {accuracy:.4f}')
    return {'accuracy': accuracy, 'count': total}

def evaluate_cleaned(all_prediction):
    all_prediction['clean_answer'] = all_prediction['clean_answer'].astype(str).str.upper()
    all_prediction['gt_answers'] = all_prediction['gt_answers'].astype(str).str.upper()
    all_prediction['is_correct'] = all_prediction['clean_answer'] == all_prediction['gt_answers']
    accuracy_per_bias = all_prediction.groupby('bias_type')['is_correct'].mean().reset_index()
    accuracy_per_bias.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    print(accuracy_per_bias)
    return accuracy_per_bias
