"""File interface for the recovered conv_ai/eval.py BOLD implementation."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from llmbias.evaluation.bold_original import ID_PATTERNS, analyze

MODEL = "unitary/toxic-bert"
SOURCE_SHA256 = "4146d6ba82be0e3766cf9f557d40a1252ffb646e4513b89a8a82695122c8c288"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name):
    try:
        return version(name)
    except Exception:
        return "unavailable"


def summarize_domains(domain_df):
    """Equal mean over the script's extracted groups; agrees with saved Table 14 row.

    This reporting step is new: the source script only exports the group table.
    Its 'domain' keys are request-ID categories, not the five BOLD domains.
    """
    return {"avg_sentiment": float(domain_df["avg_sentiment"].mean()),
            "avg_toxicity": float(domain_df["avg_toxicity"].mean()),
            "count": int(domain_df["count"].sum()), "groups": len(domain_df),
            "aggregation": "unweighted_mean_of_original_script_groups"}


def evaluate(input_path, output_dir, response_format="openai", id_format="bold",
             model_revision=None, device=None):
    """Run original raw-text scoring; loads unitary/toxic-bert on invocation only.

    response_format/id_format select alternatives explicitly present in eval.py.
    With revision/device omitted, pipeline arguments match the recovered script.
    A model download may be needed; no provider generation API is called.
    """
    if response_format not in {"openai", "anthropic", "deepseek"} or id_format not in ID_PATTERNS:
        raise ValueError("Invalid response_format or id_format")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Select a fresh output directory: {output}")
    # Preserve the source's blank-line/malformed-JSON handling and source order.
    with Path(input_path).open() as stream:
        records = [json.loads(line) for line in stream]
    if not records:
        raise ValueError("No BOLD response records")
    if any(r.get("method") == "POST" and "url" in r and "body" in r and "response" not in r for r in records):
        raise ValueError("Input contains generation requests, not responses; select the response JSONL")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline
    vader = SentimentIntensityAnalyzer()
    options = {"model": MODEL, "return_all_scores": True}
    if model_revision is not None:
        options["revision"] = model_revision
    if device is not None:
        options["device"] = device
    classifier = pipeline("text-classification", **options)
    domain_df, gender_df = analyze(records, vader, classifier, response_format, id_format)
    summary = summarize_domains(domain_df)
    config = getattr(getattr(classifier, "model", None), "config", None)
    provenance = {"implementation": "recovered_conv_ai_eval", "source_sha256": SOURCE_SHA256,
                  "input_sha256": _hash(input_path), "input_file": Path(input_path).name, "input_records": len(records),
                  "response_format": response_format, "id_format": id_format,
                  "text_policy": "raw_response_no_prompt_prefix_no_anonymization",
                  "sentiment": {"implementation": "vaderSentiment", "version": _version("vaderSentiment"),
                                "score": "compound"},
                  "toxicity": {"model": MODEL, "pipeline_arguments": options,
                               "resolved_revision": getattr(config, "_commit_hash", None),
                               "score": "label.lower() == 'toxic'; fallback 0.0",
                               "transformers_version": _version("transformers"), "torch_version": _version("torch")},
                  "aggregation": summary["aggregation"],
                  "overall_reporting": "New mean of original output groups; saved LLaMA-8B row agrees with Table 14 rounding",
                  "historical_model_revision_known": False, "all_paper_rows_verified": False}
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("domain_metrics.csv", domain_df), ("gender_polarity.csv", gender_df)]:
        with (output/name).open("x", newline="") as stream:
            frame.to_csv(stream, index=False)
    for name, data in [("overall.json", summary), ("provenance.json", provenance)]:
        with (output/name).open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return summary
