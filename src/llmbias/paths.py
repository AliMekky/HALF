"""Filesystem configuration, separate from scientific configuration."""
import os
from pathlib import Path

def project_root():
    return Path(os.environ.get("LLMBIAS_ROOT", Path(__file__).resolve().parents[2]))

def legacy_root():
    return Path(os.environ.get("LLMBIAS_LEGACY_ROOT", project_root() / "legacy"))

def workspace_path(relative):
    relative = Path(relative)
    prefix = Path("repo/LLMBias")
    if relative.is_relative_to(prefix):
        return legacy_root() / relative.relative_to(prefix)
    workspace = os.environ.get("LLMBIAS_WORKSPACE")
    if workspace is None:
        raise RuntimeError("Set LLMBIAS_WORKSPACE to the original source-data workspace for external assets")
    return Path(workspace) / relative
