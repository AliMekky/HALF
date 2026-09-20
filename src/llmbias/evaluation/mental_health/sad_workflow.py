"""File inputs and output handling for the domain evaluation."""
from pathlib import Path

def evaluate_sad(gold_csv, neutral_csv, sensitive_csv, model_name, output_csv_path):
    from llmbias.evaluation.mental_health.sad import evaluate_model, LABELS
    return evaluate_model(gold_csv, neutral_csv, sensitive_csv, LABELS, model_name, output_csv_path)
