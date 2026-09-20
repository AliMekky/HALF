"""Filesystem-oriented entry points; scientific behavior lives in unchanged modules."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from llmbias.data.paths import data_root

COMMANDS = {
    "normalize-paper": "llmbias.evaluation.normalization",
    "assemble-table": "llmbias.evaluation.table_assembly",
    "evaluate-half": "llmbias.evaluation.aggregation",
    "submit-openai": "llmbias.providers.openai",
    "submit-anthropic": "llmbias.providers.anthropic",
    "submit-deepseek": "llmbias.providers.deepseek",
    "submit-deepinfra": "llmbias.providers.deepinfra",
    "parse-openai": "llmbias.processing.parsing.mcq_openai",
    "parse-anthropic": "llmbias.processing.parsing.mcq_anthropic",
    "parse-deepseek": "llmbias.processing.parsing.mcq_deepseek",
    "evaluate-accuracy": "llmbias.evaluation.medical.accuracy",
    "evaluate-medical": "llmbias.evaluation.medical.scoring",
    "prepare-legal-tensors": "llmbias.evaluation.legal.tensors",
    "evaluate-legal": "llmbias.evaluation.legal.scoring",
    "merge-summaries": "llmbias.processing.parsing.summarization_merge",
}


def build(dataset, model, output_dir, data_root):
    """Build requests using the existing builders, prompts, and generation settings."""
    import pandas as pd
    from llmbias.processing.requests.prompts import DATASETS
    from llmbias.data.registry import functions

    name = dataset.split("/")[-1]
    if name not in functions:
        raise ValueError(f"Dataset {name} is outside the paper scope. Choose from {sorted(functions)}")
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
    if name in ("medbullets", "djinni", "CAMS", "SAD", "movielens"):
        requests, neutral = functions[name](df, model, name, template)
    elif name in DATASETS:
        requests, neutral = functions[name](df, model, name, template), None
    else:
        raise ValueError(f"Dataset {name} is not supported. Please choose from {DATASETS.keys()}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Model identifiers are scientific inputs; only their filename spelling changes.
    from urllib.parse import quote
    model_file = quote(model, safe="-_.")
    outputs = [(output / f"{model_file}_{name}.jsonl", requests)]
    if neutral is not None:
        outputs.append((output / f"{model_file}_{name}_neutral.jsonl", neutral))
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
    sub.add_parser("list", help="List paper-scoped workflows")
    sub.add_parser("coverage", help="Show located implementations and explicit task gaps")
    from llmbias._workflows import WORKFLOWS
    workflow = sub.add_parser("workflow", help="Run a notebook-derived workflow with explicit JSON configuration")
    workflow.add_argument("name", choices=sorted(WORKFLOWS))
    workflow.add_argument("--config", type=Path, help="JSON object with function argument names; omit to inspect the workflow")
    workflow.add_argument("--output", type=Path, help="Optional CSV/JSON path for the returned result; must not exist")
    create = sub.add_parser("build", help="Generate requests offline into a new output directory")
    create.add_argument("--dataset", required=True, help="e.g. medical_data/medbullets")
    create.add_argument("--model", default="gpt-4o")
    create.add_argument("--output-dir", required=True)
    create.add_argument("--data-root", type=Path, default=data_root())
    run = sub.add_parser("run", help="Run a relocated script with its original CLI")
    run.add_argument("workflow", choices=COMMANDS)
    run.add_argument("args", nargs=argparse.REMAINDER, help="Original script flags after --")
    args = parser.parse_args()
    if args.command == "list":
        for name, module in COMMANDS.items():
            print(f"{name:24} {module}")
        for name, spec in WORKFLOWS.items():
            print(f"{name:38} {spec.target} ({spec.source})")
    elif args.command == "coverage":
        from llmbias._workflows import TASK_COVERAGE
        print(json.dumps(TASK_COVERAGE, indent=2))
    elif args.command == "workflow":
        from llmbias._workflows import run as run_workflow
        if args.config is None:
            spec = WORKFLOWS[args.name]
            print(f"{spec.target}\nSource: {spec.source}\n{spec.note}")
            print("Supply --config with a JSON object matching the function arguments.")
        else:
            configuration = json.loads(args.config.read_text())
            if not isinstance(configuration, dict):
                parser.error("Workflow configuration must be a JSON object")
            run_workflow(args.name, configuration, args.output)
    elif args.command == "build":
        build(args.dataset, args.model, args.output_dir, args.data_root)
    else:
        forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
        previous_argv, previous_cwd = sys.argv, Path.cwd()
        try:
            sys.argv = [COMMANDS[args.workflow], *forwarded]
            runpy.run_module(COMMANDS[args.workflow], run_name="__main__")
        finally:
            sys.argv = previous_argv
            os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
