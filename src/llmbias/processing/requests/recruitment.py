import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_djinni_template(df, model, dataset, base_template_dic):
    string = """Candidate CV which is for {combination}:"""
    batch_result = []
    batch_result_neutral = []
    for i in range(len(df[:500])):
        # Create neutral version
        neutral_template = copy.deepcopy(base_template_dic)
        neutral_template['custom_id'] = f"request-{i}-{model}-{dataset}-neutral"
        neutral_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_RECRUITMENT
        neutral_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(resume = df.iloc[i]['cv_text'], job_description = df.iloc[i]['job_description'])
        batch_result_neutral.append(neutral_template)
        for combination in RECRUITMENT_COMBINATIONS:
            # combination = combination.replace('minor', 'young adult')
            random_name = random.choice(COMBINATION_NAMES[combination])
            temp = string.format(name='', combination=combination)
            temp_template = copy.deepcopy(base_template_dic)
            temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{combination}"
            temp_template['body']['messages'][0]['content'] =  SYSTEM_PROMPT_RECRUITMENT
            temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(resume = df.iloc[i]['cv_text'], job_description = df.iloc[i]['job_description'])
            temp_template['body']['messages'][1]['content'] = temp_template['body']['messages'][1]['content'].replace('Candidate CV:', temp)
            batch_result.append(temp_template)

    return batch_result, batch_result_neutral
