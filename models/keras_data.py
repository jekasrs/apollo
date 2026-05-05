"""
Numpy-представления реплик из того же ``samples.pkl``, что и Apollo (см. ``Dataset.padding``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics

from dataset import SAMPLES_PKL
from dataset.models.Dataset import _sample_audio_vec
from dataset.preprocess.utils import utils as dataset_utils


def utterance_feature_vector(
    sample: Any, modalities: str
) -> np.ndarray:
    t = torch.as_tensor(sample.embeddings, dtype=torch.float32)
    a = _sample_audio_vec(sample)
    if modalities == "a":
        feat = a
    elif modalities == "t":
        feat = t
    else:
        raise ValueError(f"Unknown modalities: {modalities}")

    return feat.numpy().astype(np.float32)


def samples_to_xy(
    samples: list, modalities: str
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(
        [utterance_feature_vector(s, modalities) for s in samples]
    )
    y = np.array([int(s.label) for s in samples], dtype=np.int64)
    return X, y


def load_preprocess_splits(pkl_path: Optional[Path] = None):
    path = Path(pkl_path) if pkl_path else Path(SAMPLES_PKL)
    data = dataset_utils.load_pickle(path)
    train = data["train"]
    dev = data["dev"]
    test = data["test"]
    return train, dev, test


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    return X[..., np.newaxis]


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    return X[:, np.newaxis, :]


def print_metrics_like_apollo(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_to_idx: Dict[str, int],
    test_loss: Optional[float] = None,
) -> None:
    golds = np.asarray(y_true)
    pred_labels = np.asarray(y_pred)
    weighted_f1 = metrics.f1_score(
        golds, pred_labels, average="weighted", zero_division=0
    )
    acc = metrics.accuracy_score(golds, pred_labels)
    print(
        metrics.classification_report(
            golds,
            pred_labels,
            target_names=list(label_to_idx.keys()),
            digits=4,
            zero_division=0,
        )
    )
    if test_loss is not None:
        print(
            f"Accuracy: {acc:.4f}  Weighted F1: {weighted_f1:.4f}  Loss (sum/n_batches): {test_loss:.4f}"
        )
    else:
        print(f"Accuracy: {acc:.4f}  Weighted F1: {weighted_f1:.4f}")


def save_run_meta(path: Path, meta: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_run_meta(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
