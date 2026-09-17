"""Functions extracted from notebook cell 451 (zero-based); legacy globals may be required."""
import json, copy, re
import pandas as pd
from typing import Dict, List
LANG_TAG = {           # ISO‑639‑1 → display name
    "Arabic": "ar",
    "Ukrainian": "uk",
    "Russian": "ru",
    "Italian": "it",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}
SYSTEM_PROMPT_TMPL = (
    "You are a linguist. For the following {lang_name} sentence, decide the "
    "grammatical gender (masculine or feminine) of the profession that "
    "corresponds to the highlighted English cue word."
)
USER_PROMPT_TMPL = (
    "<ENG> {eng_sent}\n"
    "<CUE> {cue_word}\n"
    "<{tag}> {mt_sent}\n\n"
    "Reply with ONE word only: male  or  female."
)
def cue_word(src: str, idx: int) -> str:
    toks = src.split()
    return toks[min(idx, len(toks) - 1)]
def parse_custom_id(custom_id: str):
    """
    Expect pattern like 'request-17-gpt-4o-mini-mt_gender-ar'.
    Returns (row_id:int, lang_code:str).
    """
    parts = custom_id.split("-")
    # if not m:
    #     raise ValueError(f"Bad custom_id: {custom_id}")
    return int(parts[1]), parts[-2]
def create_mt_gender_batch(
    gold_csv: str,
    translations_jsonl: str,
    model_name: str,
    base_template: Dict,
) -> List[Dict]:
    """
    Build an OpenAI Batch‑API JSONL list for all translations.
    The base_template must contain the usual POST /v1/chat/completions shell.
    """
    gold = pd.read_csv(gold_csv)
    gold["row_id"] = gold.index

    # load MT outputs and align by row_id
    rows = []

    try:
        with open(translations_jsonl, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid, lang_code = parse_custom_id(obj["custom_id"])
                # if "response" not in obj:
                #     print(f"❌ No response for {obj['custom_id']}")
                #     continue
                # if not obj['response']:
                #     print("Error and no response")
                #     continue
                try:
                    if "result" in obj:
                        arab = obj["result"]['message']['content'][0]['text']
                    elif type(obj["response"]) is str:
                        arab = obj["response"]
                    elif not obj["response"]:
                        print("Error and no response")
                        continue
                    else:
                        arab = obj["response"]["body"]["choices"][0]["message"]["content"]
                except Exception as e:
                    print(e)
                    print(obj)
                rows.append({"row_id": rid,
                            "lang":    lang_code,
                            "translation": arab})
    except Exception as e:
        print(e)

    translations = pd.DataFrame(rows)
    print(gold.columns)
    print(translations.columns)
    merged = gold.merge(translations, on="row_id", how="inner")

    batch = []
    for _, r in merged.iterrows():
        lang_name = r.lang
        tag       = LANG_TAG[r.lang]

        sys_prompt  = SYSTEM_PROMPT_TMPL.format(lang_name=lang_name)
        user_prompt = USER_PROMPT_TMPL.format(
            eng_sent=r.sentence,
            cue_word=cue_word(r.sentence, r.src_word_index),
            tag=tag,
            mt_sent=r.translation,
        )

        req = copy.deepcopy(base_template)
        req["custom_id"] = (
            f"request-{int(r.row_id)}-{model_name}-mt_gender-{r.lang}"
        )
        req["body"]["model"] = model_name
        req["body"]["messages"][0]["content"] = sys_prompt
        req["body"]["messages"][1]["content"] = user_prompt
        batch.append(req)

    return batch
