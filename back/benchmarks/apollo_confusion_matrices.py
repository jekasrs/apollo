"""
Строит матрицы ошибок (confusion matrix) на test для сохранённых чекпоинтов Apollo.

Пример (из корня репозитория):
  PYTHONPATH=. python3 benchmarks/apollo_confusion_matrices.py

По умолчанию три чекпоинта modality-suite с early stopping:
  a/t/at → results/apollo_meld_suite_{a,t,at}_es/model.pt

Вывод в каталог (по умолчанию results/apollo_meld_suite_confusion/):
  {tag}_confusion_counts.csv / .png   — абсолютные числа; строки = истина, столбцы = предсказание
  {tag}_confusion_row_norm.csv / .png — нормализация по строке (доли при данном истинном классе)
  {tag}_confusion.json               — то же в JSON + accuracy / weighted_f1 / macro_f1

Отключить картинки: --no-plots

Строки и столбцы упорядочены как в EMOTION_MAP (neutral, surprise, fear, …).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

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

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

DEFAULT_SUITE_CHECKPOINTS: tuple[tuple[str, Path], ...] = (
    ("a", Path("results/apollo_meld_suite_a_es/model.pt")),
    ("t", Path("results/apollo_meld_suite_t_es/model.pt")),
    ("at", Path("results/apollo_meld_suite_at_es/model.pt")),
)


def _emotion_labels_ordered() -> list[str]:
    return list(dataset_constants.EMOTION_MAP.keys())


def _build_test_dataset(pkl: Path, ckpt: dict) -> Dataset:
    data = dataset_utils.load_pickle(str(pkl.resolve()))
    train_samples = data["train"]
    s0 = train_samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        pause_mu, pause_std = float(mu), float(std)
    else:
        pause_mu, pause_std = dataset_utils.compute_pause_norm_stats(train_samples)
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        _LOG.warning("pickle: нет pause_norm_mu/std — пересчёт по train")

    use_pause, modalities, dpb, modality_feature_dim = _dataset_options_from_ckpt(ckpt)

    return Dataset(
        data["test"],
        dialogues_per_batch=dpb,
        modalities=modalities,
        modality_feature_dim=modality_feature_dim,
        pause_mu=pause_mu,
        pause_std=pause_std,
        augment=False,
        use_pause=use_pause,
    )


def _confusion_row_normalized(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    out = np.zeros_like(cm, dtype=np.float64)
    np.divide(cm, rs, out=out, where=rs > 0)
    return out


def _save_csv_matrix(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "," + ",".join(labels)
    lines = [header]
    for i, row_label in enumerate(labels):
        lines.append(row_label + "," + ",".join(str(int(x)) if cm.dtype.kind == "i" else f"{x:.6g}" for x in cm[i]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_confusion_png(
    path: Path,
    cm: np.ndarray,
    names: list[str],
    title: str,
    *,
    values_format: str | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(ax=ax, cmap="Blues", values_format=values_format, colorbar=True, im_kw={"vmin": 0})
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_one(
    tag: str,
    checkpoint: Path,
    pkl: Path,
    out_dir: Path,
    device,
    *,
    save_plots: bool,
    plot_dpi: int,
) -> dict:
    ckpt_path = checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"[{tag}] нет файла чекпоинта: {ckpt_path}")

    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = load_train_checkpoint(bundle, device)
    test_ds = _build_test_dataset(pkl, bundle)

    gold, logits = collect_logits_and_golds(
        model, test_ds, device, desc=f"test logits [{tag}]"
    )
    pred = np.argmax(logits, axis=1)
    gold = gold.astype(np.int64).ravel()

    labels = np.arange(len(_emotion_labels_ordered()))
    cm = metrics.confusion_matrix(gold, pred, labels=labels)
    cm_rn = _confusion_row_normalized(cm)

    names = _emotion_labels_ordered()
    acc = float(metrics.accuracy_score(gold, pred))
    wf1 = float(metrics.f1_score(gold, pred, average="weighted", zero_division=0))
    mf1 = float(metrics.f1_score(gold, pred, average="macro", zero_division=0))

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = out_dir / f"{tag}_confusion_counts.csv"
    rownorm_csv = out_dir / f"{tag}_confusion_row_norm.csv"
    json_path = out_dir / f"{tag}_confusion.json"

    _save_csv_matrix(counts_csv, cm.astype(np.int64), names)
    _save_csv_matrix(rownorm_csv, cm_rn, names)

    payload = {
        "tag": tag,
        "checkpoint": str(ckpt_path),
        "samples_pkl": str(pkl.resolve()),
        "labels_rows_true_cols_pred": names,
        "confusion_counts": cm.tolist(),
        "confusion_row_normalized": cm_rn.tolist(),
        "accuracy": acc,
        "weighted_f1": wf1,
        "macro_f1": mf1,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if save_plots:
        counts_png = out_dir / f"{tag}_confusion_counts.png"
        rownorm_png = out_dir / f"{tag}_confusion_row_norm.png"
        _save_confusion_png(
            counts_png,
            cm.astype(np.int64),
            names,
            title=f"MELD test — {tag} (числа), acc={acc:.3f}, wF1={wf1:.3f}",
            values_format="d",
            dpi=plot_dpi,
        )
        _save_confusion_png(
            rownorm_png,
            cm_rn,
            names,
            title=f"MELD test — {tag} (доля по строке / истинный класс)",
            values_format=".2f",
            dpi=plot_dpi,
        )
        _LOG.info("[%s] PNG: %s, %s", tag, counts_png.name, rownorm_png.name)

    _LOG.info(
        "[%s] test accuracy=%.4f weighted_f1=%.4f macro_f1=%.4f → %s",
        tag,
        acc,
        wf1,
        mf1,
        out_dir,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Матрицы ошибок Apollo на test для набора чекпоинтов")
    parser.add_argument(
        "--samples-pkl",
        type=Path,
        default=None,
        help="samples.pkl (по умолчанию dataset.SAMPLES_PKL / env APOLLO_SAMPLES_PKL)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/apollo_meld_suite_confusion"),
        help="Куда сохранить CSV, JSON и PNG",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не сохранять PNG (только CSV/JSON)",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=160,
        help="Разрешение PNG (dpi)",
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        metavar="TAG:PATH",
        help="Пары тег:чекпоинт; если не указано — три modality-suite по умолчанию",
    )
    args = parser.parse_args()

    pkl = args.samples_pkl.resolve() if args.samples_pkl is not None else Path(SAMPLES_PKL).resolve()
    device = arguments_and_constants.DEVICE

    if args.pairs:
        checkpoints: list[tuple[str, Path]] = []
        for raw in args.pairs:
            if ":" not in raw:
                raise SystemExit(f"Ожидался формат TAG:PATH, получено: {raw!r}")
            tag, path = raw.split(":", 1)
            checkpoints.append((tag.strip(), Path(path.strip())))
    else:
        checkpoints = list(DEFAULT_SUITE_CHECKPOINTS)

    save_plots = not args.no_plots
    for tag, path in checkpoints:
        run_one(
            tag,
            path,
            pkl,
            args.out_dir,
            device,
            save_plots=save_plots,
            plot_dpi=args.plot_dpi,
        )

    _LOG.info("Готово: %s", args.out_dir.resolve())


if __name__ == "__main__":
    main()
