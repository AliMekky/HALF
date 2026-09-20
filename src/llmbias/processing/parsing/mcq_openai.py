import argparse
import json
import csv
import re
from pathlib import Path

def extract_metadata(entry):
    custom_id = entry.get("custom_id", "")

    # Extract index
    index_match = re.search(r"request-(\d+)", custom_id)
    index = int(index_match.group(1)) if index_match else None

    # Extract answer
    answer = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # Extract gender and ethnicity from the end of the custom_id
    parts = custom_id.split("-")
    gender = parts[-1] if len(parts) >= 2 else None
    ethnicity = parts[-2] if len(parts) >= 2 else None

    # Only assign if they look valid (avoid capturing model name etc.)
    if gender not in {"male", "female"}:
        gender = None
    if ethnicity.lower() not in {"asian", "arab", "western"}:  # add more as needed
        ethnicity = None

    return {
        "index": index,
        "answer": answer,
        "gender": gender,
        "ethnicity": ethnicity
    }

def main(input_path):
    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            metadata = extract_metadata(entry)
            if metadata["index"] is not None:
                results.append(metadata)

    # Define output CSV path
    output_path = Path(input_path).with_suffix(".csv")

    # Save to CSV
    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["index", "answer", "gender", "ethnicity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Saved {len(results)} entries to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MCQ answers and metadata from a JSONL file.")
    parser.add_argument("--input_path", type=str, help="Path to the input JSONL file.")
    args = parser.parse_args()
    main(args.input_path)
