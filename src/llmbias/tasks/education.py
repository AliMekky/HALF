import copy
import re
import random
from typing import Dict, List
from llmbias.prompts import *
from llmbias.interventions.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_education_template(
        df,                  
        model: str,
        dataset: str,
        base_template_dic: Dict,
    ):
    # roles = ['teacher', 'student', 'generation']
    roles = ['teacher', 'student']

    batch_result = []

    for i in range(len(df)):
        for role in roles:
            if role == 'teacher':
                system_prompt = SYSTEM_PROMPT_EDUCATION_TEACHER
            elif role == 'student':
                system_prompt = SYSTEM_PROMPT_EDUCATION_STUDENT
            elif role == 'generation':
                system_prompt = SYSTEM_PROMPT_EDUCATION_GENERATION

            if role == 'generation' and re.search(r"\d$", df.iloc[i]['topic']):
                # Skip the generation role for rows with a number at the end
                continue

            if re.search(r"\d$", df.iloc[i]['topic']):
                topic = ' '.join(df.iloc[i]['topic'].split('-')[:-1])
            else:
                topic = df.iloc[i]['topic']

            for combination in EDUCATION_COMBINATION:
                temp_template = copy.deepcopy(base_template_dic)
                combination_id = combination.replace(' ', '-')
                temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{role}-{combination_id }"
                temp_template['body']['messages'][0]['content'] =  system_prompt
                # if role == 'generation':
                #     temp_template['body']['messages'][1]['content'] = DATASETS[dataset + '_' + role]['zero_shot_prompt'].format(combination = combination, topic = topic)
                # else:
                temp_template['body']['messages'][1]['content'] = DATASETS[dataset][f'zero_shot_prompt_{role}'].format(combination = combination, topic = topic, option1 = df.iloc[i]['col_1'], option2 = df.iloc[i]['col_2'], option3 = df.iloc[i]['col_3'], option4 = df.iloc[i]['col_4'], option5 = df.iloc[i]['col_5'])
                batch_result.append(temp_template)

    return batch_result

def create_education_admission_template(
        df,                  
        model: str,
        dataset: str,
        base_template_dic: Dict,
    ):
    batch_result = []

    for i in range(len(df[:500])):

        for combination in EDUCATION_COMBINATION:
            prompt = df.iloc[i]['prompt_neutral']
            prompt = prompt.replace('The student', f'The {combination} student')
            temp_template = copy.deepcopy(base_template_dic)
            temp_template['custom_id'] = f"request-{i}-{model}-{dataset}"
            temp_template['body']['messages'][0]['content'] =  SYSTEM_PROMPT_EDUCATION_ADMISSION
            temp_template['body']['messages'][1]['content'] = prompt
            batch_result.append(temp_template)

    return batch_result
