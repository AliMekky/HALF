import os, json, time, argparse, pathlib
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()                   # looks for .env in current folder

# ---------- Helpers ---------------------------------------------------------
import re
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

def load_openai_batch(jsonl_path, claude_model, default_max_tok=1024):
    """
    Convert an OpenAI-format JSONL file to Claude batch Requests.
    • Moves any {'role':'system'} message to top-level `system`
    • Rewrites 'gpt-4o' → 'claude' inside custom_id strings
    """
    requests = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            entry     = json.loads(line)
            body      = entry.get("body", {})
            messages  = body.get("messages", [])

            # --- custom_id ------------------------------------------------
            custom_id = entry.get("custom_id", f"req-{idx}")
            # print("custom_id:", custom_id)

            # ▲ replace gpt-4o / gpt4o / GPT-4O variants with 'claude'
            custom_id = re.sub(r"gpt[-_]?4o", "claude", custom_id, flags=re.I)
            custom_id = custom_id.replace(" ", "-")  # replace spaces with dashes
            # --- extract system vs chat messages --------------------------
            sys_msgs  = [m["content"] for m in messages if m["role"] == "system"]
            chat_msgs = [m for m in messages if m["role"] != "system"]
            system_prompt = "\n\n".join(sys_msgs) if sys_msgs else None

            # --- build Claude request -------------------------------------
            req = Request(
                custom_id = custom_id,
                params    = MessageCreateParamsNonStreaming(
                    model       = claude_model,
                    messages    = chat_msgs,
                    system      = system_prompt,
                    max_tokens  = body.get("max_tokens", default_max_tok),
                    temperature = body.get("temperature", 0.0)
                )
            )
            requests.append(req)
    return requests


def poll_batch(client, batch_id, every=30):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"[{time.strftime('%H:%M:%S')}] Batch {batch_id}: {status}")
        if status == "ended":
            return batch
        time.sleep(every)

def save_results(client, batch_id, out_path):
    out_path = pathlib.Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in client.messages.batches.results(batch_id):
            # quickest: get a JSON string straight from the model
            f.write(entry.to_json() + "\n")          # <- 🔑 change
    print(f"✅ Results written to {out_path.resolve()}")

# ---------- Main ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Submit an OpenAI-style JSONL batch to Claude")
    parser.add_argument("--input",  required=True,
                        help="Path to OpenAI-format JSONL file")
    parser.add_argument("--output", default="claude_batch_results.jsonl",
                        help="Where to store Claude responses")
    parser.add_argument("--model",  default="claude-sonnet-4-20250514",
                        help="Claude model ID")
    parser.add_argument("--poll",   type=int, default=30,
                        help="Polling interval (s)")
    args = parser.parse_args()

    # 1. Load and convert the batch
    reqs = load_openai_batch(args.input, args.model)
    print(f"📄 Prepared {len(reqs)} requests for Claude")

    # 2. Submit
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    batch  = client.messages.batches.create(requests=reqs)
    print(f"🚀 Submitted. Batch ID: {batch.id}")

    # 3. Poll until it finishes
    poll_batch(client, batch.id, args.poll)

    # 4. Stream and save results
    save_results(client, batch.id, args.output)

if __name__ == "__main__":
    main()
