from llmbias.paths import workspace_path
import json
import re
from collections import defaultdict
from pathlib import Path


# ✅ Paths to your English source files
source_file_pro = str(workspace_path('en_pro.txt'))
source_file_anti = str(workspace_path('en_anti.txt'))

# ✅ Load English sentences (only column 3 = index 2)
def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().split("\t")[2] for line in f.readlines()]

source_sentences_pro = load_sentences(source_file_pro)
source_sentences_anti = load_sentences(source_file_anti)

# ✅ Combine into one full list
num_pro = len(source_sentences_pro)
num_anti = len(source_sentences_anti)
print(f"✅ Loaded {num_pro} PRO sentences and {num_anti} ANTI sentences (total {num_pro + num_anti}).")

# ✅ Load response file
with open(str(workspace_path('repo/LLMBias/experiments/translation_data/output_files/gpt-4.1-2025-04-14_mt_gender.jsonl')), "r", encoding="utf-8") as f:
    response_data = [json.loads(line) for line in f]

# ✅ Organize translations by (language, group)
bitexts = defaultdict(list)

for entry in response_data:
    custom_id = entry["custom_id"]

    try:
        # Extract index
        index_match = re.search(r'request-(\d+)-', custom_id)
        if not index_match:
            print(f"❌ Could not find index in {custom_id}")
            continue
        index = int(index_match.group(1))

        # Determine group (pro/anti) based on index
        if index < num_pro:
            group = "pro"
            source = source_sentences_pro[index]
        else:
            group = "anti"
            source = source_sentences_anti[index - num_pro]

        # Extract language
        lang_match = re.search(r'mt_gender-([A-Za-z]+)-', custom_id)
        lang = lang_match.group(1) if lang_match else "Unknown"

        # Extract translation output
        translation = entry["response"]["body"]["choices"][0]["message"]["content"].strip()

        # ✅ Compose line in the required format
        output_line = f"{source} ||| {translation}"

        key = f"{lang}_{group}"
        bitexts[key].append(output_line)

    except Exception as e:
        print(f"Error parsing {custom_id}: {e}")


# ✅ Write bitext files for each (language, group)
output_dir = Path("bitext_outputs")
output_dir.mkdir(exist_ok=True)

for key, lines in bitexts.items():
    lang, group = key.split("_")
    output_path = output_dir / f"en-{lang}_{group}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"✅ Saved {output_path}")
