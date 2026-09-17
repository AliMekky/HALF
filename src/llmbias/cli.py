"""Filesystem-oriented entry points; scientific behavior lives in unchanged modules."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from llmbias.paths import legacy_root

COMMANDS = {
    "create-batch": "llmbias.pipeline.create_batch",
    "submit-openai": "llmbias.providers.openai_batch",
    "submit-anthropic": "llmbias.providers.anthropic_batch",
    "submit-deepseek": "llmbias.providers.deepseek",
    "submit-deepinfra": "llmbias.providers.deepinfra",
    "parse-openai": "llmbias.parsing.openai_mcq",
    "parse-anthropic": "llmbias.parsing.claude_mcq",
    "parse-deepseek": "llmbias.parsing.deepseek_mcq",
    "evaluate-accuracy": "llmbias.evaluation.accuracy",
    "evaluate-medical": "llmbias.evaluation.medical",
    "prepare-legal-tensors": "llmbias.evaluation.legal_tensors",
    "evaluate-legal": "llmbias.evaluation.legal",
    "merge-summaries": "llmbias.evaluation.summarization",
}


def build(dataset, model, output_dir, data_root):
    """Build requests using the existing builders, prompts, and generation settings."""
    import pandas as pd
    from llmbias.prompts import DATASETS
    from llmbias.registry import functions

    name = dataset.split("/")[-1]
    df = pd.read_csv(Path(data_root) / f"{dataset}.csv")
    template = {
        "custom_id": None, "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": model, "messages": [
            {"role": "system", "content": None},
            {"role": "user", "content": None},
        ]},
    }
    if "o4" not in model.lower():
        template["body"]["temperature"] = 0.6
    if name in ("medbullets", "djinni", "CAMS", "SAD", "dreaddit", "movielens"):
        requests, neutral = functions[name](df, model, name, template)
    elif name in DATASETS:
        requests, neutral = functions[name](df, model, name, template), None
    else:
        raise ValueError(f"Dataset {name} is not supported. Please choose from {DATASETS.keys()}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    outputs = [(output / f"{model}_{name}.jsonl", requests)]
    if neutral is not None:
        outputs.append((output / f"{model}_{name}_neutral.jsonl", neutral))
    # New filesystem guard only: never overwrite an existing run accidentally.
    for path, _ in outputs:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Select a fresh output directory.")
    for path, records in outputs:
        with path.open("w") as stream:
            for record in records:
                json.dump(record, stream)
                stream.write("\n")
        print(f"Wrote {len(records)} requests to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List unchanged legacy workflows")
    create = sub.add_parser("build", help="Generate requests offline into a new output directory")
    create.add_argument("--dataset", required=True, help="e.g. medical_data/medbullets")
    create.add_argument("--model", default="gpt-4o")
    create.add_argument("--output-dir", required=True)
    create.add_argument("--data-root", type=Path, default=legacy_root())
    run = sub.add_parser("run", help="Run a relocated script with its original CLI")
    run.add_argument("workflow", choices=COMMANDS)
    run.add_argument("args", nargs=argparse.REMAINDER, help="Original script flags after --")
    args = parser.parse_args()
    if args.command == "list":
        for name, module in COMMANDS.items():
            print(f"{name:24} {module}")
    elif args.command == "build":
        build(args.dataset, args.model, args.output_dir, args.data_root)
    else:
        forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
        previous_argv, previous_cwd = sys.argv, Path.cwd()
        try:
            # The original batch builder resolves data relative to experiments/.
            if args.workflow == "create-batch":
                os.chdir(legacy_root() / "experiments")
            sys.argv = [COMMANDS[args.workflow], *forwarded]
            runpy.run_module(COMMANDS[args.workflow], run_name="__main__")
        finally:
            sys.argv = previous_argv
            os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
