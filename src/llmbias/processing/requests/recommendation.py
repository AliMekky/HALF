import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_rec_system_template(df, model, dataset, base_template_dic):
    batch_result = []
    batch_result_neutral = []

    for i in range(len(df)):
        # Create neutral version
        anchor_type = df.iloc[i]['anchor_type']
        neutral_template = copy.deepcopy(base_template_dic)
        neutral_template['custom_id'] = f"request-{i}-{model}-{dataset}-{anchor_type}-neutral"
        neutral_template['body']['messages'][0]['content'] = SYSTEM_PROMPT_REC_SYSTEM
        neutral_template['body']['messages'][1]['content'] = DATASETS[dataset]['neutral_prompt'].format(genres = df.iloc[i]['genre_phrase'], years = df.iloc[i]['year_range'], movies = df.iloc[i]['anchor_movies'])
        batch_result_neutral.append(neutral_template)

        for combination in COMBINATIONS:
            # anchor_type = df.iloc[i]['anchor_type']
            temp_template = copy.deepcopy(base_template_dic)
            combination_id = combination.replace(' ', '-')
            temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{anchor_type}-{combination_id}"
            temp_template['body']['messages'][0]['content'] =  SYSTEM_PROMPT_REC_SYSTEM
            temp_template['body']['messages'][1]['content'] = DATASETS[dataset]['zero_shot_prompt'].format(combination = combination, genres = df.iloc[i]['genre_phrase'], years = df.iloc[i]['year_range'], movies = df.iloc[i]['anchor_movies'])
            batch_result.append(temp_template)

    return batch_result, batch_result_neutral
