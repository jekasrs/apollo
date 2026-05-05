"""
Серия прогонов train.py: фиксированно --modalities at --use-pause, подбор остального.

Запуск из корня репозитория:
  PYTHONPATH=. python3 models/apollo/trainings/search_at_pause.py

Пишет results/apollo_meld_at_pause_search/results.json, копии прогона в run_*.pt,
лучший чекпоинт → results/apollo_meld_at_pause_search/best_at_pause_search.pt (по dev score).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAIN = ROOT / "models" / "apollo" / "trainings" / "train.py"
EVAL = ROOT / "models" / "apollo" / "trainings" / "eval.py"
OUT_DIR = ROOT / "results" / "apollo_meld_at_pause_search"


def _run_ckpt_path(idx: int, name: str) -> Path:
    rid = f"apollo_meld_pause_{idx:02d}_{name}"
    return ROOT / "results" / rid / "model.pt"


@dataclass
class RunConfig:
    name: str
    extra: list[str]  # argv tokens after base


def _parse_eval_output(text: str) -> dict[str, float | None]:
    out = {
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    }
    m = re.search(r"Accuracy:\s+([0-9.]+)", text)
    if m:
        out["accuracy"] = float(m.group(1))
    m = re.search(r"Macro F1:\s+([0-9.]+)", text)
    if m:
        out["macro_f1"] = float(m.group(1))
    m = re.search(r"Weighted F1:\s+([0-9.]+)", text)
    if m:
        out["weighted_f1"] = float(m.group(1))
    return out  # type: ignore


# Подобранные итерации: LR, gamma, dev-metric, дубли, focal vs CE, beta
DEFAULT_RUNS: list[RunConfig] = [
    RunConfig("lr2e-4_g1.5", []),
    RunConfig("lr1e-4_g1.5", ["--learning-rate", "1e-4", "--focal-gamma", "1.5"]),
    RunConfig("lr3e-4_g1.5", ["--learning-rate", "3e-4", "--focal-gamma", "1.5"]),
    RunConfig("lr4e-4_g1.5", ["--learning-rate", "4e-4", "--focal-gamma", "1.5"]),
    RunConfig("lr2e-4_g1.0", ["--focal-gamma", "1.0"]),
    RunConfig("lr2e-4_g2.0", ["--focal-gamma", "2.0"]),
    RunConfig("lr2e-4_g2.5", ["--focal-gamma", "2.5"]),
    RunConfig("dev_macro", ["--dev-metric", "macro_f1"]),
    RunConfig("dup_fd1", ["--train-dup-fear-disgust", "1"]),
    RunConfig("no_focal_sm", ["--no-focal", "--label-smoothing", "0.08"]),
    RunConfig("beta_0.99", ["--class-weight-beta", "0.99"]),
    RunConfig("lr1e-4_g2.5", ["--learning-rate", "1e-4", "--focal-gamma", "2.5"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid at+pause → train + eval, JSON summary")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Только печать команд, без train/eval",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "results.json",
        help="Куда писать JSON",
    )
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for i, run in enumerate(DEFAULT_RUNS, start=1):
        base = [
            sys.executable,
            str(TRAIN),
            "--modalities",
            "at",
            "--use-pause",
            "--run-id",
            f"apollo_meld_pause_{i:02d}_{run.name}",
            *run.extra,
        ]
        if args.dry_run:
            print(" ".join(base))
            continue

        print(f"\n=== [{i}/{len(DEFAULT_RUNS)}] {run.name} ===\n" + " ".join(base) + "\n", flush=True)
        subprocess.run(base, cwd=str(ROOT), env=env, check=True)

        CKPT = _run_ckpt_path(i, run.name)
        if not CKPT.is_file():
            raise FileNotFoundError(CKPT)

        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        best_dev = ckpt.get("best_dev_score")
        if best_dev is None:
            best_dev = ckpt.get("best_dev_f1")
        best_ep = ckpt.get("best_epoch")
        run_ckpt = OUT_DIR / f"run_{i:02d}_{run.name}.pt"
        shutil.copy2(CKPT, run_ckpt)

        p = subprocess.run(
            [sys.executable, str(EVAL), "--checkpoint", str(CKPT)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        text = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            print(text, file=sys.stderr)
            raise SystemExit(f"eval failed: {run.name}")

        ev = _parse_eval_output(text)
        row = {
            "idx": i,
            "name": run.name,
            "extra_args": run.extra,
            "best_dev_score": float(best_dev) if best_dev is not None else None,
            "dev_select_metric": ckpt.get("dev_select_metric"),
            "best_epoch": int(best_ep) if best_ep is not None else None,
            "checkpoint": str(run_ckpt.relative_to(ROOT)),
            "test": ev,
        }
        rows.append(row)
        print(
            f"  → dev best={row['best_dev_score']!r} ep={row['best_epoch']} "
            f"test wF1={ev.get('weighted_f1')} acc={ev.get('accuracy')}\n",
            flush=True,
        )

    if args.dry_run:
        return

    def _key_dev(r: dict) -> float:
        v = r.get("best_dev_score")
        if v is None:
            return -1.0
        return float(v)

    by_dev = max(rows, key=_key_dev)
    by_test_wf1 = max(
        rows,
        key=lambda r: float((r.get("test") or {}).get("weighted_f1") or -1.0),
    )
    out_payload = {
        "fixed": {"modalities": "at", "use_pause": True},
        "runs": rows,
        "best_by_dev": by_dev,
        "best_by_test_weighted_f1": by_test_wf1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    best_path = OUT_DIR / f"run_{by_dev['idx']:02d}_{by_dev['name']}.pt"
    if best_path.is_file():
        shutil.copy2(best_path, OUT_DIR / "best_at_pause_search.pt")
    print("Сводка (dev):", by_dev["name"], "score=", by_dev.get("best_dev_score"), "test wF1=", (by_dev.get("test") or {}).get("weighted_f1"))
    print("Сводка (test wF1):", by_test_wf1["name"], "wF1=", (by_test_wf1.get("test") or {}).get("weighted_f1"))
    print("JSON:", args.out)
    print("best_at_pause_search.pt ←", best_path.name)


if __name__ == "__main__":
    main()