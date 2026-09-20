import argparse
import csv
import re
from pathlib import Path

COMBINATIONS = [
    ("western", "male"),
    ("western", "female"),
    ("arab", "male"),
    ("arab", "female"),
    ("asian", "male"),
    ("asian", "female"),
]

NUM_GROUPS = 1740 // 6  # Expecting 290 groups

def parse_conversations(text):
    # Returns a list of all blocks like: (conv_number, conv_text)
    blocks = re.findall(r"=== Conversation (\d+) ===\s*(.*?)=== DeepSeek Response ===\s*([A-E]?)\s*={3,}", text, re.DOTALL)
    responses = {}
    for conv_num_str, _, answer in blocks:
        conv_num = int(conv_num_str)
        responses[conv_num] = answer if answer else None
    return responses

def group_conversations(responses):
    results = []
    for group_index in range(NUM_GROUPS):
        for i, (eth, gen) in enumerate(COMBINATIONS):
            conv_id = group_index * 6 + i + 1  # Conversation IDs start from 1
            answer = responses.get(conv_id, None)
            results.append({
                "index": group_index,
                "ethnicity": eth,
                "gender": gen,
                "answer": answer
            })
    return results

def main(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    responses = parse_conversations(text)
    data = group_conversations(responses)

    output_path = Path(input_path).with_suffix(".csv")
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "ethnicity", "gender", "answer"])
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Processed {len(data)} entries into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the input .txt file")
    args = parser.parse_args()
    main(args.input_path)
