"""Explicit file/configuration boundaries around the existing analysis functions."""
from pathlib import Path


def evaluate_recruitment(neutral_csv, sensitive_csv, model_name):
    import pandas as pd
    from llmbias.analysis.recruitment import add_reference_columns, summarise
    rows = add_reference_columns(pd.read_csv(neutral_csv), pd.read_csv(sensitive_csv))
    combo = summarise(rows, ["gender", "ethnicity"], "gender_ethnicity")
    gender = summarise(rows, ["gender"], "gender")
    ethnic = summarise(rows, ["ethnicity"], "ethnicity")
    out = pd.concat([combo, gender, ethnic], ignore_index=True)
    out["model"] = model_name
    return out


def evaluate_sad(gold_csv, neutral_csv, sensitive_csv, model_name, output_csv_path):
    from llmbias.analysis.sad import evaluate_model, LABELS
    return evaluate_model(gold_csv, neutral_csv, sensitive_csv, LABELS, model_name, output_csv_path)


def evaluate_education(csv_dir_paths, output_dir, task="ranking", bootstrap=5000):
    from llmbias.analysis.education import compute_bias_by_role
    norm_by_role, bias_by_role = compute_bias_by_role(csv_dir_paths, task=task, bootstrap=bootstrap)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_by_role["teacher"].to_csv(output_dir / "teacher_bias.csv", index=False)
    bias_by_role["student"].to_csv(output_dir / "student_bias.csv", index=False)
    return bias_by_role


def report_cams_intersectional(neutral_csv, sensitive_csv, gold_csv, output_dir, save_heatmaps=True):
    """Relocate the historical relative report paths into an explicit directory.

    This adapter changes process working directory temporarily; run it serially.
    """
    import os
    from llmbias.analysis.cams_intersectional import analyze
    inputs = [str(Path(path).resolve()) for path in (neutral_csv, sensitive_csv, gold_csv)]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(destination)
        return analyze(*inputs, out_group_csv="per_group_metrics.csv",
                       out_examples="qual_examples.jsonl", save_heatmaps=save_heatmaps)
    finally:
        os.chdir(previous)
