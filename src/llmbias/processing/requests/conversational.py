import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_bold_template(
    df,
    model: str,
    dataset: str,
    base_template_dic: Dict,
):
    batch_result = []

    for i in range(len(df)):
        text = df.iloc[i]['prompt']  
        category = df.iloc[i]['category']  
        subcategory = df.iloc[i]['subcategory']  

        temp_template = copy.deepcopy(base_template_dic)
        temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{category}-{subcategory}"
        temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_BOLD
        temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(text = text)

        batch_result.append(temp_template)

    return batch_result

def create_bbq_template(
    df,
    model: str,
    dataset: str,
    base_template_dic: Dict,
):
    batch_result = []

    for i in range(len(df)):
        example_id = df.iloc[i]['example_id']
        question_index = df.iloc[i]['question_index']
        category = df.iloc[i]['category']  
        context = df.iloc[i]['context']  
        question = df.iloc[i]['question'] 
        ans0 = df.iloc[i]['ans0'] 
        ans1 = df.iloc[i]['ans1'] 
        ans2 = df.iloc[i]['ans2'] 


        temp_template = copy.deepcopy(base_template_dic)
        temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{category}-{example_id}-{question_index}"
        temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_BBQ
        temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(context = context, question = question, option1 = ans0, option2 = ans1, option3 = ans2)

        batch_result.append(temp_template)

    return batch_result
