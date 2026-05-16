"""Пути относительно корня репозитория (папка apollo/)."""

from __future__ import annotations

from pathlib import Path

# back/models/apollo/utils/repo_paths.py → parents[4] = корень репозитория (apollo/)
_REPO_ROOT = Path(__file__).resolve().parents[4]


def repo_root() -> Path:
    return _REPO_ROOT


def results_dir() -> Path:
    return _REPO_ROOT / "results"


def default_run_results_dir(run_id: str = "apollo_meld_at_r01") -> Path:
    return results_dir() / run_id


def docs_dir() -> Path:
    return _REPO_ROOT / "documentation"
