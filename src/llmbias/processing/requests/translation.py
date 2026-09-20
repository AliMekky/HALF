import copy
import re
import random
from typing import Dict, List
from llmbias.processing.requests.prompts import *
from llmbias.processing.preprocessing.demographics import GENDERS, swap_gender, remove_gender_mentions, add_ethnicity_to_gender

def create_translation_template(df, model, dataset, base_template_dic):
    language_map = {
        "ar": "Arabic",
        "uk": "Ukrainian",
        "ru": "Russian",
        "it": "Italian",
        "fr": "French",
        "es": "Spanish",
        "de": "German"
    }


    batch_result = []

    for i in range(len(df)):
        sentence = df.iloc[i]['sentence']  # assumes your dataframe has a 'sentence' column
        type = df.iloc[i]['type']  # assumes your dataframe has a 'type' column

        for lang_code, lang_name in language_map.items():
            temp_template = copy.deepcopy(base_template_dic)

            prompt = DATASETS[dataset]['zero_shot_prompt'].format(
                sentence=sentence,
                target_language=lang_name
            )

            temp_template['custom_id'] = f"request-{i}-{model}-{dataset}-{lang_name.replace(' ', '_')}-{type}"
            temp_template['body']['messages'][0]['content'] = "You are a professional translator."
            temp_template['body']['messages'][1]['content'] = prompt

            batch_result.append(temp_template)

    return batch_result
