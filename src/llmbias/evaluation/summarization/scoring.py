"""Run the preserved summary_bias implementation in its own environment.

No scorer is reimplemented here. Upstream parsing may require spaCy model
resources; upstream scoring retains its own dependencies and NLTK resource use.
"""
import os
from pathlib import Path
import subprocess
import sys

from llmbias.data.paths import project_root


def commands(input_path, output_dir, upstream_root=None, python=None):
    root = Path(upstream_root) if upstream_root else project_root() / "third_party/summary_bias"
    root = root.resolve()
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    parsed_dir = output_dir / "parsed_summaries"
    python = python or sys.executable
    return root, [
        [python, "-m", "summarybias.evaluate.parse", str(input_path), "--dir", str(parsed_dir)],
        [python, str(root / "scripts/evaluate_gender_file.py"), str(parsed_dir / input_path.name)],
    ]


def evaluate(input_path, output_dir, upstream_root=None, python=None):
    root, calls = commands(input_path, output_dir, upstream_root, python)
    if not (root / "summarybias").is_dir():
        raise FileNotFoundError(f"Run scripts/setup_summary_bias.py or set upstream_root: {root}")
    output_dir = Path(output_dir).resolve()
    (output_dir / "parsed_summaries").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    # The working directory supplies upstream relative word-list/data paths.
    subprocess.run(calls[0], cwd=root, env=env, check=True)
    result = subprocess.run(calls[1], cwd=root, env=env, check=True, text=True, capture_output=True)
    (output_dir / "gender_scores.txt").write_text(result.stdout)
    (output_dir / "gender_scores.stderr.txt").write_text(result.stderr)
    return {"scores": str(output_dir / "gender_scores.txt")}
