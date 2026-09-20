"""File inputs and output handling for the domain evaluation."""
from pathlib import Path

def evaluate_education(csv_dir_paths, output_dir, task="ranking", bootstrap=5000):
    from llmbias.evaluation.education.scoring import compute_bias_by_role
    norm_by_role, bias_by_role = compute_bias_by_role(csv_dir_paths, task=task, bootstrap=bootstrap)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_by_role["teacher"].to_csv(output_dir / "teacher_bias.csv", index=False)
    bias_by_role["student"].to_csv(output_dir / "student_bias.csv", index=False)
    return bias_by_role
