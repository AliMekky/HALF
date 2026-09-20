"""Filesystem configuration for paper inputs, outputs, and external tools."""
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get("LLMBIAS_ROOT", Path(__file__).resolve().parents[3]))


def data_root():
    return Path(os.environ.get("LLMBIAS_DATA_ROOT", project_root() / "data" / "processed"))


def runs_root():
    return Path(os.environ.get("LLMBIAS_RUNS_ROOT", project_root() / "runs"))
