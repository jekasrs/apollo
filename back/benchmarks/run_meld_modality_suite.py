#!/usr/bin/env python3
"""
Серия прогонов на MELD (один ``samples.pkl``): CNN, DNN, LSTM для модальностей a, t, at;
Apollo с GNN heterogeneous (RGCN) и homogeneous (GCN первый слой).

Для каждого прогона сохраняются веса и ``metrics.json`` на **test**: accuracy, weighted F1,
F1 по каждой эмоции.

Пример (из каталога ``back/`` с ``PYTHONPATH=.``):

  PYTHONPATH=. python benchmarks/run_meld_modality_suite.py --samples-pkl dataset/preprocess/samples/samples.pkl

Опции:
  --epochs-keras 15 --epochs-apollo 40 --apollo-use-pause
  --only keras | apollo | cnn dnn apollo ...
  --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("a", "t", "at")
KERAS_SCRIPTS = {
    "cnn": ROOT / "models" / "cnn" / "train.py",
    "dnn": ROOT / "models" / "dnn" / "train.py",
    "lstm": ROOT / "models" / "lstm" / "train.py",
}
TRAIN_PY = ROOT / "models" / "apollo" / "trainings" / "train.py"
EVAL_PY = ROOT / "models" / "apollo" / "trainings" / "eval.py"


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT)}


def _write_summary(base: Path, rows: list[dict[str, Any]], emotion_cols: list[str]) -> None:
    lines = [
        "# MELD — сводная таблица (test)",
        "",
        "По каждой эмоции: **F1** (one-vs-rest для класса). По всему test: **accuracy**, **weighted F1**.",
        "",
        "**С гетеризацией (рёбер):** модель строки ``apollo_rgcn`` — RGCN. **Без:** ``apollo_gcn`` — первый слой GCN без типов рёбер.",
        "",
    ]
    current_mod: str | None = None
    for r in sorted(rows, key=lambda x: (x["modalities"], x["model"])):
        m = str(r["modalities"])
        if m != current_mod:
            current_mod = m
            hdr = (
                "| model | accuracy | weighted_f1 | "
                + " | ".join(f"`{e}` F1" for e in emotion_cols)
                + " |"
            )
            sep = "|" + "|".join(["---"] * (3 + len(emotion_cols))) + "|"
            lines.extend([f"## Модальность `{m}`", "", hdr, sep])
        pe = r.get("per_emotion") or {}
        ef = [str(round(float(pe.get(em, {}).get("f1", 0.0)), 4)) for em in emotion_cols]
        lines.append(
            f"| {r['model']} | {float(r['accuracy']):.4f} | {float(r['weighted_f1']):.4f} | "
            + " | ".join(ef)
            + " |"
        )
    lines.append("")
    (base / "SUMMARY_TABLES.md").write_text("\n".join(lines), encoding="utf-8")

    merged = Path(base / "all_metrics.json")
    merged.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Полная сетка MELD по моделям и модальностям")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "meld_modality_suite",
        help="Корень артефактов (подкаталоги per-run + SUMMARY_TABLES.md)",
    )
    ap.add_argument("--samples-pkl", type=Path, required=True, help="pickle train/dev/test (MELD)")
    ap.add_argument("--epochs-keras", type=int, default=20)
    ap.add_argument("--epochs-apollo", type=int, default=50)
    ap.add_argument(
        "--apollo-use-pause",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Пауза в Apollo (по умолчанию True, как типичный мультимодальный прогон)",
    )
    ap.add_argument(
        "--apollo-early-stop",
        type=int,
        default=5,
        help="Early stopping patience для Apollo (0 = выкл.)",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Подмножество: cnn dnn lstm apollo_het apollo_hom (пусто = всё)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = args.out_dir.resolve()
    base.mkdir(parents=True, exist_ok=True)

    from dataset.preprocess.utils import constants as c

    emotion_cols = list(c.EMOTION_MAP.keys())

    only = None
    if args.only:
        only = frozenset(args.only)

    def want(name: str) -> bool:
        return only is None or name in only

    rows: list[dict[str, Any]] = []

    for mod in MODALITIES:
        for keras_name, script in KERAS_SCRIPTS.items():
            key = keras_name
            if not want(key):
                continue
            tag = f"{keras_name}_{mod}"
            run_dir = base / tag
            mj = run_dir / "metrics.json"
            cmd = [
                sys.executable,
                str(script),
                "--modalities",
                mod,
                "--epochs",
                str(args.epochs_keras),
                "--out-dir",
                str(run_dir),
                "--samples-pkl",
                str(args.samples_pkl.resolve()),
                "--export-metrics-json",
                str(mj),
            ]
            print("→", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)
                rows.append(json.loads(mj.read_text(encoding="utf-8")))

        for het, row_name, only_tags in [
            (True, "apollo_rgcn", ("apollo_het", "apollo_rgcn")),
            (False, "apollo_gcn", ("apollo_hom", "apollo_gcn")),
        ]:
            if only is not None and not any(t in only for t in only_tags):
                continue
            mode = "heterogeneous" if het else "homogeneous"
            rid = f"meld_suite_apollo_{mod}_{'rgcn' if het else 'gcn'}"
            run_dir = base / f"apollo_{mod}_{'rgcn' if het else 'gcn'}"
            run_dir.mkdir(parents=True, exist_ok=True)
            mj = run_dir / "metrics.json"

            pause_args = ["--use-pause"] if args.apollo_use_pause else ["--no-pause"]

            tc = [
                sys.executable,
                str(TRAIN_PY),
                "--modalities",
                mod,
                "--run-id",
                rid,
                "--epochs",
                str(args.epochs_apollo),
                "--early-stopping-patience",
                str(args.apollo_early_stop),
                "--samples-pkl",
                str(args.samples_pkl.resolve()),
                "--gnn-edge-mode",
                mode,
                *pause_args,
            ]
            print("→", " ".join(tc), flush=True)
            if not args.dry_run:
                subprocess.run(tc, cwd=str(ROOT), env=_env(), check=True)
                ckpt = ROOT / "results" / rid / "model.pt"
                ev_cmd = [
                    sys.executable,
                    str(EVAL_PY),
                    "--checkpoint",
                    str(ckpt),
                    "--samples-pkl",
                    str(args.samples_pkl.resolve()),
                    "--export-metrics-json",
                    str(mj),
                ]
                print("→", " ".join(ev_cmd), flush=True)
                subprocess.run(ev_cmd, cwd=str(ROOT), env=_env(), check=True)
                row = json.loads(mj.read_text(encoding="utf-8"))
                if "per_emotion" not in row and "per_emotion_f1" in row:
                    row["per_emotion"] = {
                        em: {"f1": float(v)} for em, v in row["per_emotion_f1"].items()
                    }
                row["model"] = row_name
                row["modalities"] = mod
                row["apollo_checkpoint"] = str(ckpt)
                rows.append(row)

    if args.dry_run:
        print("Dry run: таблицы не собраны.")
        return

    _write_summary(base, rows, emotion_cols)
    print(f"Готово: {base / 'SUMMARY_TABLES.md'}\nОбъединённый JSON: {base / 'all_metrics.json'}")


if __name__ == "__main__":
    main()
