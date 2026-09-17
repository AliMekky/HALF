"""Historical notebook cells [450]; structural extraction, unchanged scientific logic."""
import json, re, copy
import pandas as pd
from typing import Dict, List
SYSTEM_PROMPT_TMPL = 'You are a linguist. In the following {lang_name} sentence, decide the grammatical gender (masculine or feminine) of the profession that corresponds to the highlighted English cue word.'
USER_PROMPT_TMPL = '<ENG> {eng_sent}\n<CUE> {cue_word}\n<{lang_tag}> {mt_sent}\n\nReply with ONE word only: male  or  female.'
LANG_MAP = {'Arabic': ('Arabic', 'ARB'), 'French': ('French', 'FRA'), 'Spanish': ('Spanish', 'SPA'), 'German': ('German', 'DEU')}

def cue_word(src_sentence: str, src_idx: int) -> str:
    toks = src_sentence.split()
    return toks[min(src_idx, len(toks) - 1)]

def parse_custom_id(cid: str):
    """
    Extract (row_id:int, tgt_lang:str) from custom_id like
    'request-42-gpt-4.1-mini-mt_gender-Arabic-pro'
    """
    m = re.match('request-(\\d+)-.+-mt_gender-([^-]+)', cid)
    if not m:
        raise ValueError(f'Bad custom_id pattern: {cid}')
    return (int(m.group(1)), m.group(2))

def create_mt_gender_batch(gold_csv: str, translations_jsonl: str, model_name: str, base_tmpl: Dict) -> List[Dict]:
    """
    Return a list of Batch‑API request dicts –\xa0one per translation sentence.
    """
    gold = pd.read_csv(gold_csv)
    gold['row_id'] = gold.index
    rows = []
    with open(translations_jsonl, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            (rid, lang) = parse_custom_id(obj['custom_id'])
            sent = obj['response']['body']['choices'][0]['message']['content']
            rows.append({'row_id': rid, 'lang': lang, 'translation': sent})
    trans = pd.DataFrame(rows)
    df = gold.merge(trans, on='row_id', how='inner')
    batch = []
    for (_, r) in df.iterrows():
        (lang_name, lang_tag) = LANG_MAP.get(r.lang, (r.lang, r.lang[:3].upper()))
        sys_prompt = SYSTEM_PROMPT_TMPL.format(lang_name=lang_name)
        user_prompt = USER_PROMPT_TMPL.format(eng_sent=r.sentence, cue_word=cue_word(r.sentence, r.src_word_index), lang_tag=lang_tag, mt_sent=r.translation)
        req = copy.deepcopy(base_tmpl)
        req['custom_id'] = f'request-{int(r.row_id)}-{model_name}-mt_gender-{r.lang}'
        req['body']['model'] = model_name
        req['body']['messages'][0]['content'] = sys_prompt
        req['body']['messages'][1]['content'] = user_prompt
        batch.append(req)
    return batch
