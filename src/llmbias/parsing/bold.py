"""Join saved BOLD responses to the frozen prompt table without positional guessing.

This is a new adapter, not recovered historical scoring code. Text assembly and
anonymization are explicit because the publication does not identify those
choices for the saved chat-completion responses.
"""
import hashlib
import json
from pathlib import Path
import re

DOMAINS = {
    "profession_prompt.json": "profession",
    "gender_prompt.json": "gender",
    "race_prompt.json": "race",
    "religious_ideology_prompt.json": "religious_ideology",
    "political_ideology_prompt.json": "political_ideology",
}


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def response_text(record):
    """Recognize actual saved OpenAI-compatible and Anthropic envelopes."""
    if record.get("error"):
        raise ValueError("Provider returned an error")
    if "result" in record:
        result = record["result"]
        if result.get("type", "succeeded") != "succeeded":
            raise ValueError(f"Anthropic result: {result.get('type')}")
        body = result["message"]
        if body.get("stop_reason") in {"max_tokens", "refusal"}:
            raise ValueError("Incomplete or refused Anthropic response")
        text = "".join(block["text"] for block in body["content"] if block.get("type") == "text")
    elif "response" in record:
        envelope = record["response"]
        if envelope.get("status_code", 200) != 200:
            raise ValueError(f"Provider HTTP status: {envelope.get('status_code')}")
        body = envelope["body"]
        choice = body["choices"][0]
        if choice.get("finish_reason", "stop") != "stop":
            raise ValueError(f"Incomplete response: {choice.get('finish_reason')}")
        if choice["message"].get("refusal"):
            raise ValueError("Provider refused the request")
        text = choice["message"]["content"]
    else:
        raise ValueError("Unknown BOLD response envelope; provide the original batch response")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty/non-text completion")
    return text, body.get("model")


def assemble_text(prompt, response, text_mode):
    if text_mode == "response":
        return response
    if text_mode == "full_text":
        # Preserve a full echoed prompt once; otherwise join prompt + continuation.
        if response.lstrip().startswith(prompt.strip()):
            return response
        return prompt.rstrip() + " " + response.lstrip()
    raise ValueError("text_mode must explicitly be 'response' or 'full_text'")


def anonymize(text, terms, domain):
    """BOLD §3.3 replacement tokens; caller supplies reviewed entity spans/terms.

    Matching literal terms with word boundaries is this adapter's implementation;
    the authors did not publish their original entity-extraction implementation.
    """
    if not isinstance(terms, list) or any(not isinstance(t, str) or not t.strip() for t in terms):
        raise ValueError("Each entity-map value must be a list of nonempty literal terms")
    if not terms:
        return text
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)) + r")(?!\w)"
    return re.sub(pattern, "Person" if domain in {"gender", "race"} else "XYZ", text, flags=re.IGNORECASE)


def process_file(input_path, dataset_path, output_path, model_name, text_mode,
                 anonymization, entities_path=None, require_complete=True,
                 metadata_overrides_path=None, id_policy="exact", batch_path=None):
    """Convert one model's saved responses; fail on errors, duplicates or bad joins.

    anonymization='explicit' requires a JSON object mapping each CSV row index to
    its reviewed list of terms; 'none' is an explicitly unmasked alternative.
    """
    import pandas as pd
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name is required; request IDs may name a different model")
    if anonymization not in {"explicit", "none"}:
        raise ValueError("anonymization must be 'explicit' or 'none'")
    if (anonymization == "explicit") != (entities_path is not None):
        raise ValueError("Supply entities_path exactly when anonymization='explicit'")
    entities = json.loads(Path(entities_path).read_text()) if entities_path else {}
    if not isinstance(entities, dict):
        raise ValueError("Entity map must be a JSON object keyed by dataset row index")
    dataset = pd.read_csv(dataset_path, keep_default_na=False)
    required = {"prompt", "file", "category", "subcategory"}
    if not required.issubset(dataset.columns):
        raise ValueError(f"Dataset requires {sorted(required)}")
    if id_policy not in {"exact", "anthropic_sanitized"}:
        raise ValueError("id_policy must be 'exact' or 'anthropic_sanitized'")
    overrides = json.loads(Path(metadata_overrides_path).read_text()) if metadata_overrides_path else {}
    if not isinstance(overrides, dict):
        raise ValueError("metadata_overrides_path must contain an object keyed by row index")
    for key, override in overrides.items():
        if not key.isdigit() or int(key) >= len(dataset) or not isinstance(override, dict) or not override.get("source"):
            raise ValueError("Each metadata override needs a valid row index and source provenance")
        if set(override) - required - {"source"}:
            raise ValueError("Unexpected metadata override field")
        for column in required & set(override):
            dataset.loc[int(key), column] = override[column]
    batches = {}
    if batch_path:
        for request in read_jsonl(batch_path):
            match = re.match(r"^request-(\d+)-", request.get("custom_id", ""))
            if not match or int(match.group(1)) in batches:
                raise ValueError("Invalid/duplicate request index in batch file")
            batches[int(match.group(1))] = request
    source_hashes = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
                     for key, path in {"dataset": dataset_path, "responses": input_path,
                         "entities": entities_path, "batch": batch_path,
                         "metadata_overrides": metadata_overrides_path}.items()}
    records, seen, source_models = [], set(), set()
    for response in read_jsonl(input_path):
        cid = response.get("custom_id", "")
        match = re.match(r"^request-(\d+)-", cid)
        if not match:
            raise ValueError(f"Malformed request ID: {cid!r}")
        index = int(match.group(1))
        if index in seen or index >= len(dataset):
            raise ValueError(f"Duplicate/out-of-range BOLD row index: {index}")
        seen.add(index)
        row = dataset.iloc[index]
        suffix = f"-{row['category']}-{row['subcategory']}"
        if id_policy == "anthropic_sanitized":
            suffix = re.sub(r"[^a-zA-Z0-9_-]", "", suffix)
        if not cid.endswith(suffix):
            raise ValueError(f"Request metadata does not match dataset row {index}: {cid}")
        domain = DOMAINS.get(row["file"])
        if domain is None:
            raise ValueError(f"Unknown BOLD domain file: {row['file']}")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"Empty BOLD source prompt at row {index}; resolve against the saved request")
        if batch_path:
            if index not in batches:
                raise ValueError(f"Missing batch request for row {index}")
            request = batches[index]
            if not request["custom_id"].endswith(f"-{row['category']}-{row['subcategory']}"):
                raise ValueError(f"Batch metadata does not match row {index}")
            if request["body"]["messages"][1]["content"] != row["prompt"]:
                raise ValueError(f"Batch prompt does not match row {index}")
        raw, source_model = response_text(response)
        if source_model:
            source_models.add(source_model)
        text = assemble_text(row["prompt"], raw, text_mode)
        if anonymization == "explicit":
            if str(index) not in entities:
                raise ValueError(f"Missing reviewed anonymization terms for row {index}")
            text = anonymize(text, entities[str(index)], domain)
        records.append({"custom_id": cid, "index": index, "model": model_name,
                        "source_model": source_model, "domain": domain,
                        "category": row["category"], "subcategory": row["subcategory"],
                        "prompt": row["prompt"], "response": raw, "text": text,
                        "text_sha256": text_hash(text), "text_mode": text_mode,
                        "anonymization": anonymization, "id_policy": id_policy,
                        "metadata_override": overrides.get(str(index)),
                        "conversion_sources": source_hashes})
    if not records:
        raise ValueError("No BOLD responses found")
    if len(source_models) > 1:
        raise ValueError(f"Mixed model responses: {sorted(source_models)}")
    if require_complete and seen != set(range(len(dataset))):
        raise ValueError(f"Missing {len(dataset) - len(seen)} BOLD responses")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for record in sorted(records, key=lambda r: r["index"]):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"responses": len(records), "expected": len(dataset), "missing": len(dataset)-len(seen),
            "source_models": sorted(source_models), "output": str(target)}
