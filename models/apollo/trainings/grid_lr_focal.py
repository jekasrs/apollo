"""
Небольшой grid по learning rate и focal gamma. Запуск из корня репозитория:

  PYTHONPATH=. python3 models/apollo/trainings/grid_lr_focal.py
  PYTHONPATH=. python3 models/apollo/trainings/grid_lr_focal.py --modalities a --epochs 2 --early-stopping-patience 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

# Сетка 3×2 = 6 прогонов
GRID_LR = [1e-4, 2e-4, 4e-4]
GRID_FOCAL_GAMMA = [1.0, 2.0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid: LR × focal gamma → train.py")
    parser.add_argument(
        "--modalities",
        default="at",
        choices=["a", "t", "at"],
        help="Модальности для всех прогонов",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--use-pause", action="store_true", help="Пауза (если не указано — из constants USE_PAUSE)")
    g.add_argument("--no-pause", action="store_true", help="Без канала паузы")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Передать в train (по умолчанию — из constants)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Передать в train",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/apollo_meld_grid_lr_focal/summary.json"),
        help="JSON с метриками по комбинациям",
    )
    args = parser.parse_args()

    root = _repo_root()
    train_script = root / "models" / "apollo" / "trainings" / "train.py"
    env = {**os.environ, "PYTHONPATH": str(root)}

    results: list[dict] = []
    for lr in GRID_LR:
        for gamma in GRID_FOCAL_GAMMA:
            cmd = [
                sys.executable,
                str(train_script),
                "--modalities",
                args.modalities,
                "--learning-rate",
                str(lr),
                "--focal-gamma",
                str(gamma),
            ]
            if args.use_pause:
                cmd.append("--use-pause")
            elif args.no_pause:
                cmd.append("--no-pause")
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.early_stopping_patience is not None:
                cmd.extend(["--early-stopping-patience", str(args.early_stopping_patience)])

            rid = f"apollo_meld_grid_lr{lr}_g{gamma}".replace(".", "_")
            cmd.extend(["--run-id", rid])
            print("→", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(root), env=env, check=True)

            ckpt_path = root / "results" / rid / "model.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_score = ckpt.get("best_dev_score")
            if best_score is None:
                best_score = ckpt.get("best_dev_f1")
            row = {
                "learning_rate": lr,
                "focal_gamma": gamma,
                "best_dev_f1": float(ckpt.get("best_dev_f1", 0.0) or 0.0)
                if ckpt.get("best_dev_f1") is not None
                else None,
                "best_dev_score": float(best_score) if best_score is not None else None,
                "best_epoch": int(ckpt.get("best_epoch", 0) or 0)
                if ckpt.get("best_epoch") is not None
                else None,
            }
            results.append(row)
            print(
                f"  lr={lr} gamma={gamma} → best dev score={row['best_dev_score']!r} @ epoch {row['best_epoch']}\n",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid_lr": GRID_LR,
        "grid_focal_gamma": GRID_FOCAL_GAMMA,
        "modalities": args.modalities,
        "runs": results,
    }
    def _score(r):
        s = r.get("best_dev_score")
        if s is not None:
            return float(s)
        return float(r["best_dev_f1"] or -1.0)

    best = max(results, key=_score)
    payload["best"] = best
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Сохранено: {args.out}")
    print(
        f"Лучшее по dev: lr={best['learning_rate']} gamma={best['focal_gamma']} → score={best.get('best_dev_score')!r} (legacy best_dev_f1={best.get('best_dev_f1')!r})"
    )


if __name__ == "__main__":
    main()
