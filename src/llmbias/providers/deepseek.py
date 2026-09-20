import json
import argparse
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def process_batch_file(input_file_path, output_file_path, api_key, model_type, retry_indices=None):
    try:
        # Initialize DeepSeek client
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat" if model_type == "chat" else "deepseek-reasoner"

        with open(input_file_path, "r") as input_file:
            lines = input_file.readlines()

        with open(output_file_path, "w") as output_file:
            for i, line in enumerate(lines):
                if retry_indices is not None and i not in retry_indices:
                    continue

                try:
                    entry = json.loads(line)
                    messages = entry["body"]["messages"]

                    conversation = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages if m["role"] in ["system", "user", "assistant"]
                    ]

                    if not conversation:
                        raise ValueError("No valid messages found in conversation")

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        stream=False,
                        temperature=0.7,
                        max_tokens=2000
                    )

                    response_content = response.choices[0].message.content

                    output_file.write(f"=== Conversation {i} ===\n")
                    for msg in conversation:
                        output_file.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    output_file.write("\n=== DeepSeek Response ===\n")
                    output_file.write(f"{response_content}\n\n")
                    output_file.write("=" * 50 + "\n\n")

                    print(f"Processed conversation {i} successfully")

                except json.JSONDecodeError as e:
                    error_msg = f"Error decoding JSON on line {i}: {e}"
                    print(error_msg)
                    output_file.write(f"ERROR: {error_msg}\n\n")
                except KeyError as e:
                    error_msg = f"Missing expected key in JSON on line {i}: {e}"
                    print(error_msg)
                    output_file.write(f"ERROR: {error_msg}\n\n")
                except Exception as e:
                    error_msg = f"Error processing line {i}: {e}"
                    print(error_msg)
                    output_file.write(f"ERROR: {error_msg}\n\n")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
    except IOError as e:
        print(f"Error handling files: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSeek batch processing.")
    parser.add_argument("--input_file", help="Path to input JSONL batch file")
    parser.add_argument("--output_file", help="Path to output result file")
    parser.add_argument("--model", choices=["chat", "reasoner"], required=True, help="Model type to use")
    parser.add_argument("--retry_indices", help="JSON file or comma-separated list of indices to retry (optional)")

    args = parser.parse_args()

    # Load API key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        print("Missing DEEPSEEK_API_KEY in environment")
        sys.exit(1)

    # Parse retry indices
    retry_indices = None
    if args.retry_indices:
        if os.path.exists(args.retry_indices):
            with open(args.retry_indices, "r") as f:
                retry_indices = json.load(f)
        else:
            retry_indices = list(map(int, args.retry_indices.split(",")))

    process_batch_file(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        api_key=DEEPSEEK_API_KEY,
        model_type=args.model,
        retry_indices=retry_indices
    )
