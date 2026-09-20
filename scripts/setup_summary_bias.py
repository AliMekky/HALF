"""Fetch the original scorer and restore its recorded resource-loading change."""
import argparse
from pathlib import Path
import subprocess

REVISION = "bac43fbea0a30ad5f97328ce4bfc7ff2f23cf7f8"
URL = "https://github.com/julmaxi/summary_bias.git"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("third_party/summary_bias"))
    args = parser.parse_args()
    target = args.destination.resolve()
    if target.exists():
        parser.error(f"Destination already exists; refusing to replace it: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", URL, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", REVISION], check=True)
    patch = Path(__file__).resolve().parent / "patches/summary-bias-resources.patch"
    subprocess.run(["git", "-C", str(target), "apply", str(patch)], check=True)
    print(f"Scorer ready: {target}\nSet upstream_root to this path in your workflow config.")


if __name__ == "__main__":
    main()
