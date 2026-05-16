#!/usr/bin/env python3
"""
Прогон DNN / LSTM / CNN на **классическом** ``samples.pkl`` (Word2Vec или FastText + MFCC или mel-спектр).

1. Собирает до четырёх pickle (если ещё нет), вызывая ``dataset/preprocess/preprocess_classical.py``.
2. Для каждого pickle и модальностей ``a``, ``t``, ``at`` запускает три Keras-скрипта с ``--export-metrics-json``.

Пример::

  PYTHONPATH=. python benchmarks/run_classical_keras_suite.py --epochs-keras 15

Уже готовые признаки можно передать явно::

  PYTHONPATH=. python benchmarks/run_classical_keras_suite.py \\
    --skip-preprocess \\
    --pkl-ft-mfcc path.pkl --pkl-ft-mel path2.pkl ...
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


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT)}


def _run(cmd: list[str], dry: bool) -> None:
    print("+", " ".join(cmd))
    if dry:
        return
    subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)


def _default_pkl(tag: str) -> Path:
    return ROOT / "dataset" / "preprocess" / "samples" / f"samples_classical_{tag}.pkl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Keras-бейзлайны на классических признаках MELD")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "classical_keras_suite")
    ap.add_argument("--epochs-keras", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true", help="Не вызывать preprocess_classical.py")
    ap.add_argument("--pkl-ft-mfcc", type=Path, default=None)
    ap.add_argument("--pkl-ft-mel", type=Path, default=None)
    ap.add_argument("--pkl-w2v-mfcc", type=Path, default=None)
    ap.add_argument("--pkl-w2v-mel", type=Path, default=None)
    args = ap.parse_args()

    py = sys.executable
    pre = ROOT / "dataset" / "preprocess" / "preprocess_classical.py"

    combos: dict[str, tuple[str, str]] = {
        "fasttext_mfcc": ("fasttext", "mfcc"),
        "fasttext_melspectrogram": ("fasttext", "melspectrogram"),
        "word2vec_mfcc": ("word2vec", "mfcc"),
        "word2vec_melspectrogram": ("word2vec", "melspectrogram"),
    }
    pkl_map: dict[str, Path] = {
        "fasttext_mfcc": args.pkl_ft_mfcc or _default_pkl("fasttext_mfcc"),
        "fasttext_melspectrogram": args.pkl_ft_mel or _default_pkl("fasttext_melspectrogram"),
        "word2vec_mfcc": args.pkl_w2v_mfcc or _default_pkl("word2vec_mfcc"),
        "word2vec_melspectrogram": args.pkl_w2v_mel or _default_pkl("word2vec_melspectrogram"),
    }

    if not args.skip_preprocess:
        for tag, (tb, ab) in combos.items():
            out = pkl_map[tag]
            if out.is_file():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    py,
                    str(pre),
                    "--text",
                    tb,
                    "--audio",
                    ab,
                    "--out",
                    str(out),
                ],
                args.dry_run,
            )

    keras_scripts = {
        "dnn": ROOT / "models" / "dnn" / "train.py",
        "lstm": ROOT / "models" / "lstm" / "train.py",
        "cnn": ROOT / "models" / "cnn" / "train.py",
    }
    modalities = ("a", "t", "at")
    rows: list[dict[str, Any]] = []

    for tag, pkl in pkl_map.items():
        if args.dry_run and not args.skip_preprocess:
            pass
        elif not pkl.is_file():
            print(f"[skip train] pickle missing: {pkl}", file=sys.stderr)
            continue
        tb, au = combos[tag]
        for km, script in keras_scripts.items():
            for mod in modalities:
                run_dir = args.out_dir / tag / km / mod
                mpath = run_dir / "metrics.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                _run(
                    [
                        py,
                        str(script),
                        "--modalities",
                        mod,
                        "--epochs",
                        str(args.epochs_keras),
                        "--samples-pkl",
                        str(pkl),
                        "--out-dir",
                        str(run_dir),
                        "--export-metrics-json",
                        str(mpath),
                    ],
                    args.dry_run,
                )
                if args.dry_run or not mpath.is_file():
                    continue
                data = json.loads(mpath.read_text(encoding="utf-8"))
                data["features_tag"] = tag
                data["text_feature"] = tb
                data["audio_feature"] = au if "mel" not in au else "melspectrogram"
                rows.append(data)

    summary = args.out_dir / "CLASSICAL_KERAS_SUMMARY.json"
    if not args.dry_run:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {summary} ({len(rows)} runs)")
    md = args.out_dir / "CLASSICAL_KERAS_TABLE.md"
    if not args.dry_run and rows:
        hdr = "| model | text | audio | modality | accuracy | weighted_f1 |\n|---|---|---|---|---|---|\n"
        lines = ["# Classical features — Keras baselines (test)", "", hdr]
        for r in sorted(
            rows,
            key=lambda x: (
                x.get("features_tag", ""),
                x.get("model", ""),
                x.get("modalities", ""),
            ),
        ):
            lines.append(
                "| {model} | {tf} | {af} | {mod} | {acc:.4f} | {wf1:.4f} |".format(
                    model=r.get("model", ""),
                    tf=r.get("text_feature", ""),
                    af=r.get("audio_feature", ""),
                    mod=r.get("modalities", ""),
                    acc=float(r["accuracy"]),
                    wf1=float(r["weighted_f1"]),
                )
            )
        lines.append("")
        md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
