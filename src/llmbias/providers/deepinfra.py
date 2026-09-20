import json
import time
import uuid
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv
load_dotenv()  # This will read from .env and set environment variables


def construct_output_path(input_path: Path, model: str) -> Path:
    input_str = str(input_path)

    # Replace 'batch_files' with 'output_files'
    updated_path = input_str.replace("batch_files", "output_files")

    # Replace model name with sanitized version
    # Match model prefix like gpt-4.1-mini-2025-04-14
    match = re.search(r"(gpt-[^_]+-\d{4}-\d{2}-\d{2})", updated_path)
    if match:
        old_model_prefix = match.group(1)
        new_model_name = model.split("/")[-1]  + '-v2' # e.g. Meta-Llama-3-8B-Instruct
        updated_path = updated_path.replace(old_model_prefix, new_model_name)
        

    return Path(updated_path)


def run_batch(input_path: Path, output_path: Path, model: str):
    # Load API key from env var
    api_key = os.getenv("DEEPINFRA_TOKEN")
    if not api_key:
        raise ValueError("Please set the DEEPINFRA_TOKEN environment variable.")

    openai = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Processing batch"):
            item = json.loads(line)
            # Extract necessary fields
            req_id = item.get("id", f"batch_req_{uuid.uuid4().hex}")
            custom_id = item.get("custom_id")
            messages = item['body']['messages']
            print(messages)

            result = {
                "id": req_id,
                "custom_id": custom_id,
                "response": None,
                "error": None
            }


            try:
                res = openai.chat.completions.create(
                    model=model,
                    messages=messages
                )

                # Build response format
                result["response"] = {
                    "status_code": 200,
                    "request_id": res.id,
                    "body": {
                        "id": res.id,
                        "object": res.object,
                        "created": int(time.time()),
                        "model": model,
                        "choices": [choice.model_dump() for choice in res.choices],
                        "usage": res.usage.model_dump() if res.usage else {},
                        "service_tier": "default",
                        "system_fingerprint": getattr(res, "system_fingerprint", None)
                    }
                }
                

            except Exception as e:
                result["error"] = str(e)

            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to the input batch JSONL file")
    parser.add_argument("--output", type=Path, help="Optional path to output file. If not provided, it will be auto-generated.")
    parser.add_argument("--model", type=str, required=True, help="Model name for DeepInfra (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    args = parser.parse_args()

    # If output is not provided, generate it
    output_path = args.output or construct_output_path(args.input, args.model)

    run_batch(args.input, output_path, args.model)
