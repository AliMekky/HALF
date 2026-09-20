"""Exercise installed CLI request generation, parsing and scoring without APIs."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory(prefix="half-smoke-") as temporary:
        root = Path(temporary)
        data = root / "data/medical_data"
        data.mkdir(parents=True)
        with (data / "medbullets.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["question", "opa", "opb", "opc", "opd", "ope", "answer_idx", "gender", "age", "age_group"])
            writer.writeheader()
            writer.writerow(dict(question="Synthetic test question", opa="One", opb="Two", opc="Three", opd="Four", ope="Five", answer_idx="A", gender="male", age=30, age_group="adult"))
        def cli(*args):
            subprocess.run([sys.executable, "-m", "llmbias.cli", *args], cwd=root, check=True)
        cli("build", "--dataset", "medical_data/medbullets", "--model", "test-model",
            "--data-root", str(root / "data"), "--output-dir", str(root / "requests"))
        request_file = root / "requests/test-model_medbullets.jsonl"
        requests = [json.loads(line) for line in request_file.read_text().splitlines()]
        assert requests, "Request builder returned no records"
        responses = root / "responses.jsonl"
        responses.write_text("".join(json.dumps({"custom_id": row["custom_id"], "response": {"body": {"choices": [{"message": {"content": "A"}}]}}}) + "\n" for row in requests))
        cli("run", "parse-openai", "--", "--input_path", str(responses))
        cli("run", "evaluate-medical", "--", "--pred", str(responses), "--gt", str(data / "medbullets.csv"), "--model_tag", "smoke")
        with responses.with_suffix(".csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == len(requests)
        outputs = list(root.rglob("accuracy_*.csv"))
        assert outputs, "Scorer produced no accuracy files"
        for output in outputs:
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            assert rows and all(float(row["Accuracy"]) == 1.0 for row in rows), output
        print("Installed CLI smoke test passed (synthetic inputs, no API calls).")


if __name__ == "__main__":
    main()
