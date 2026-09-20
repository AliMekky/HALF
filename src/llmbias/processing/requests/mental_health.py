import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_mental_health_template(df, model, dataset, base_template_dic):
    batch_result = []
    batch_result_neutral = []
    for i in range(len(df)):
        temp_template = copy.deepcopy(base_template_dic)
        temp_template['custom_id'] = f"request-{i}-{model}-{dataset}"
        temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_MENTAL_HEALTH
        temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['neutral_prompt'].format(post=df.iloc[i]['neutralized_text'])
        batch_result_neutral.append(temp_template)

        for combination in COMBINATIONS:
            temp_template = copy.deepcopy(base_template_dic)
            combination_id = combination.replace(' ', '-')
            temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{combination_id}"
            temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_MENTAL_HEALTH
            temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(post=df.iloc[i]['neutralized_text'], combination = combination)
            batch_result.append(temp_template)
    return batch_result, batch_result_neutral
