"""File inputs and output handling for the domain evaluation."""
from pathlib import Path

def report_cams_intersectional(neutral_csv, sensitive_csv, gold_csv, output_dir, save_heatmaps=True):
    """Relocate the historical relative report paths into an explicit directory.

    This adapter changes process working directory temporarily; run it serially.
    """
    import os
    from llmbias.evaluation.mental_health.intersectional import analyze
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
