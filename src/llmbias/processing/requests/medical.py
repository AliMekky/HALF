import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_med_bullets_template(df, model, dataset, base_template_dic):
    batch_result = []

    batch_result_neutral = []
    for i in range(len(df)):
            original_text = df.iloc[i]['question']
            neutral_text = remove_gender_mentions(original_text)
            temp_template = copy.deepcopy(base_template_dic)
            temp_template['custom_id'] = f"request-{i}-{model}-{dataset}"
            temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_MEDICAL

            temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(
                question=neutral_text,
                option1 = df.iloc[i]['opa'], option2 = df.iloc[i]['opb'], option3 = df.iloc[i]['opc'], option4 = df.iloc[i]['opd'], option5 = df.iloc[i]['ope'])
            batch_result_neutral.append(temp_template)        

    for i in range(len(df)):
        original_text = df.iloc[i]['question']
        if(type(df.iloc[i]['gender']) == str):
            original_gender = df.iloc[i]['gender'].lower()
        else:
            continue

        for ethnicity in MED_BULLETS_COMBINATIONS:
            for target_gender in GENDERS:
                updated_text = original_text

                # Swap gendered pronouns if switching gender
                if original_gender != target_gender:
                    updated_text = swap_gender(updated_text, from_gender=original_gender, to_gender=target_gender)

                # Replace explicit gender mention with ethnicity + gender
                updated_text = add_ethnicity_to_gender(updated_text, ethnicity)

                temp_template = copy.deepcopy(base_template_dic)
                temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{ethnicity}-{target_gender}"
                temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_MEDICAL

                temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(
                    question=updated_text,
                    option1 = df.iloc[i]['opa'], option2 = df.iloc[i]['opb'], option3 = df.iloc[i]['opc'], option4 = df.iloc[i]['opd'], option5 = df.iloc[i]['ope'])
                batch_result.append(temp_template)

    return batch_result, batch_result_neutral

def create_medical_bias_template(df, model, dataset, base_template_dic):
    batch_result = []

    for i in range(len(df)):
        bias_type = df.iloc[i]['bias_type']
        temp_template = copy.deepcopy(base_template_dic)
        temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{bias_type}"
        temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_MEDICAL
        temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(prompt = df.iloc[i]['prompt'])
        batch_result.append(temp_template)

    return batch_result
