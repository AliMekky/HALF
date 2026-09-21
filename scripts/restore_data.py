"""Rebuild the paper inputs that this repository does not redistribute.

MovieLens and OntoNotes have licences that do not allow us to redistribute
them (see data/README.md). This script rebuilds the processed files from the
original sources and checks each result against the SHA-256 recorded in
data/datasets-manifest.json.

    python scripts/restore_data.py movielens --ml20m-dir path/to/ml-20m
    python scripts/restore_data.py ontonotes --summarybias-jsonl path/to/onto-nw_gender_balanced_1.jsonl
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ANNOTATIONS = ROOT / "data" / "annotations"
MANIFEST = ROOT / "data" / "datasets-manifest.json"

def expected_sha(destination):
    for entry in json.loads(MANIFEST.read_text())["files"]:
        if entry["destination"] == destination:
            return entry["sha256"]
    return None


def write_checked(df, relative_path):
    """Write a CSV exactly as the paper pipeline did and compare its hash."""
    target = PROCESSED / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    data = buffer.getvalue().encode("utf-8")
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    expected = expected_sha(f"data/processed/{relative_path}")
    status = "matches the paper version" if digest == expected else (
        f"DOES NOT match the paper version (expected {expected}, got {digest})")
    print(f"Wrote {target.relative_to(ROOT)}: {status}")
    return digest == expected


def restore_movielens(args):
    from llmbias.processing.preprocessing.datasets import prepare_movielens
    ml = Path(args.ml20m_dir)
    ratings = pd.read_csv(ml / "ratings.csv")
    movies = pd.read_csv(ml / "movies.csv")
    return write_checked(prepare_movielens(ratings, movies), "recommendation_system/movielens.csv")


def restore_ontonotes(args):
    from llmbias.processing.preprocessing.datasets import prepare_ontonotes
    df = prepare_ontonotes(args.summarybias_jsonl)
    ids = pd.read_csv(ANNOTATIONS / "ontonotes_ids.csv")
    if not df[["sample_id", "article_id", "pair_id"]].astype(str).reset_index(drop=True).equals(
            ids.astype(str).reset_index(drop=True)):
        print("Warning: the generated instances differ from the paper's article/pair IDs.")
    return write_checked(df, "summarization_data/ontonotes.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    p = sub.add_parser("movielens", help="Rebuild the MovieLens anchors from your own ml-20m download")
    p.add_argument("--ml20m-dir", required=True, help="Folder containing ratings.csv and movies.csv")
    p.set_defaults(func=restore_movielens)

    p = sub.add_parser("ontonotes", help="Convert SummaryBias instances built from your licensed OntoNotes copy")
    p.add_argument("--summarybias-jsonl", required=True,
                   help="onto-nw_gender_balanced_1.jsonl produced by SummaryBias")
    p.set_defaults(func=restore_ontonotes)

    args = parser.parse_args()
    ok = args.func(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
