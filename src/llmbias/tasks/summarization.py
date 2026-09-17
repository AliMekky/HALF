import copy
import re
import random
from typing import Dict, List
from llmbias.prompts import *
from llmbias.interventions.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_ontonotes_template(
    df,
    model: str,
    dataset: str,
    base_template_dic: Dict,
):
    batch_result = []

    for i in range(len(df)):
        text = df.iloc[i]['text'] 
        article_id = df.iloc[i]['article_id']  
        pair_id = df.iloc[i]['pair_id']  

        temp_template = copy.deepcopy(base_template_dic)
        temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{article_id}-{pair_id}"
        temp_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_SUMMARIZATION
        temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(text = text)

        batch_result.append(temp_template)

    return batch_result
