import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_education_template(
        df,                  
        model: str,
        dataset: str,
        base_template_dic: Dict,
    ):
    roles = ['teacher', 'student']

    batch_result = []

    for i in range(len(df)):
        for role in roles:
            if role == 'teacher':
                system_prompt = SYSTEM_PROMPT_EDUCATION_TEACHER
            elif role == 'student':
                system_prompt = SYSTEM_PROMPT_EDUCATION_STUDENT

            if re.search(r"\d$", df.iloc[i]['topic']):
                topic = ' '.join(df.iloc[i]['topic'].split('-')[:-1])
            else:
                topic = df.iloc[i]['topic']

            for combination in EDUCATION_COMBINATION:
                temp_template = copy.deepcopy(base_template_dic)
                combination_id = combination.replace(' ', '-')
                temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{role}-{combination_id }"
                temp_template['body']['messages'][0]['content'] =  system_prompt
                temp_template['body']['messages'][1]['content'] = DATASETS[dataset][f'zero_shot_prompt_{role}'].format(combination = combination, topic = topic, option1 = df.iloc[i]['col_1'], option2 = df.iloc[i]['col_2'], option3 = df.iloc[i]['col_3'], option4 = df.iloc[i]['col_4'], option5 = df.iloc[i]['col_5'])
                batch_result.append(temp_template)

    return batch_result
