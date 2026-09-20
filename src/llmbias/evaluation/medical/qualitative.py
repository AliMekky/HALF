"""Historical notebook cells [429, 433, 495, 496, 497]; structural extraction, unchanged scientific logic."""
import json
import pandas as pd

def enrich_medbullets(outputs, batches, gt):
    for i in range(len(outputs)):
        outputs.at[i, 'prompt'] = batches[i]['body']['messages'][1]['content']
        outputs.at[i, 'gt_answer'] = gt.iloc[outputs.at[i, 'index']]['answer_idx']
    return outputs

def enrich_deepseek(outputs, batches):
    for i in range(len(outputs)):
        outputs.at[i, 'prompt'] = batches[i]['body']['messages'][1]['content']
    return outputs

def enrich_flip_cases(detailed, data, gt_answers):
    prompts = {}
    for item in data:
        parts = item['custom_id'].split('-')
        idx = parts[1]
        group = f'{parts[-2]}_{parts[-1]}'
        prompts[f'{idx}_{group}'] = item['body']['messages'][1]['content']
    detailed['prompts'] = detailed.apply(lambda row: prompts[f"{row['case_id']}_{row['variant']}"], axis=1)
    detailed['gt_answer'] = detailed.apply(lambda row: gt_answers[row['case_id']], axis=1)
    return detailed
