#!/usr/bin/env python3
"""
Таблица по каждой эмоции (precision, recall, F1) и общие accuracy / weighted F1 на test
для чекпоинтов modality suite (по умолчанию apollo_meld_suite_{a,t,at}_es).

Запуск из корня репозитория:
  PYTHONPATH=. python3 benchmarks/apollo_suite_emotion_table.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn import metrics

from dataset import SAMPLES_PKL
from dataset.models.Dataset import Dataset
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as dataset_utils
from models.apollo.trainings.eval import (
    _dataset_options_from_ckpt,
    collect_logits_and_golds,
    load_train_checkpoint,
)
from models.apollo.utils import constants as arguments_and_constants
from models.apollo.utils.repo_paths import repo_root

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS: list[tuple[str, Path]] = [
    ("a", Path("results/apollo_meld_suite_a_es/model.pt")),
    ("t", Path("results/apollo_meld_suite_t_es/model.pt")),
    ("at", Path("results/apollo_meld_suite_at_es/model.pt")),
]


def _pause_stats(train_samples: list) -> tuple[float, float]:
    s0 = train_samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        return float(mu), float(std)
    pm, ps = dataset_utils.compute_pause_norm_stats(train_samples)
    return float(pm), float(ps)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    root = repo_root()
    device = arguments_and_constants.DEVICE
    pkl = SAMPLES_PKL.resolve()
    data = dataset_utils.load_pickle(str(pkl))
    train_samples = data["train"]
    test_samples = data["test"]
    pause_mu, pause_std = _pause_stats(train_samples)

    emotion_names = list(dataset_constants.EMOTION_MAP.keys())
    labels = np.arange(len(emotion_names), dtype=np.int64)

    all_json: dict = {"samples_pkl": str(pkl), "runs": {}}
    md_lines: list[str] = [
        "# Apollo MELD test: метрики по эмоциям (modality suite)",
        "",
        f"`samples.pkl`: `{pkl}`",
        "",
    ]

    summary_rows: list[tuple[str, float, float]] = []

    for tag, rel in DEFAULT_CHECKPOINTS:
        ckpt_path = (root / rel).resolve()
        if not ckpt_path.is_file():
            log.warning("Пропуск (нет файла): %s", ckpt_path)
            continue

        model, meta = load_train_checkpoint(ckpt_path, device)
        raw = meta["__raw"]
        use_pause, modalities, dpb, mfd = _dataset_options_from_ckpt(raw)

        test_ds = Dataset(
            test_samples,
            dialogues_per_batch=dpb,
            modalities=modalities,
            modality_feature_dim=mfd,
            pause_mu=pause_mu,
            pause_std=pause_std,
            augment=False,
            use_pause=use_pause,
        )

        gold, logits = collect_logits_and_golds(
            model, test_ds, device, desc=f"test logits ({tag})"
        )
        gold = gold.astype(np.int64).ravel()
        pred = logits.argmax(axis=1)

        rep = metrics.classification_report(
            gold,
            pred,
            labels=labels,
            target_names=emotion_names,
            output_dict=True,
            zero_division=0,
        )
        acc = float(metrics.accuracy_score(gold, pred))
        wf1 = float(metrics.f1_score(gold, pred, average="weighted", zero_division=0))
        macro_f1 = float(metrics.f1_score(gold, pred, average="macro", zero_division=0))

        summary_rows.append((tag, acc, wf1))

        per_emotion = {}
        for name in emotion_names:
            row = rep[name]
            per_emotion[name] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1-score": float(row["f1-score"]),
                "support": int(row["support"]),
            }

        all_json["runs"][tag] = {
            "checkpoint": str(ckpt_path),
            "modalities": modalities,
            "accuracy": acc,
            "weighted_f1": wf1,
            "macro_f1": macro_f1,
            "per_emotion": per_emotion,
        }

        md_lines.append(f"## Модальность `{modalities}` (`{tag}`)")
        md_lines.append("")
        md_lines.append("| Эмоция | Precision | Recall | F1 | Support |")
        md_lines.append("|--------|-----------|--------|-----|---------|")
        for name in emotion_names:
            row = per_emotion[name]
            md_lines.append(
                f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1-score']:.4f} | {row['support']} |"
            )
        md_lines.append("")
        md_lines.append(f"**Accuracy:** {acc:.4f} **Weighted F1:** {wf1:.4f} **Macro F1:** {macro_f1:.4f}")
        md_lines.append("")

    md_lines.append("## Сводка")
    md_lines.append("")
    md_lines.append("| Run | Accuracy | Weighted F1 |")
    md_lines.append("|-----|----------|-------------|")
    for tag, acc, wf1 in summary_rows:
        md_lines.append(f"| `{tag}` | {acc:.4f} | {wf1:.4f} |")
    md_lines.append("")

    out_md = root / "results" / "apollo_meld_suite_emotion_table.md"
    out_json = root / "results" / "apollo_meld_suite_emotion_table.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json.write_text(json.dumps(all_json, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Markdown: %s", out_md)
    log.info("JSON: %s", out_json)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
