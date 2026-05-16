"""
Обучение DNN на признаках из ``samples.pkl`` (тот же пайплайн, что у Apollo).

Запуск из корня репозитория:
  PYTHONPATH=. python models/dnn/train.py --modalities at
  PYTHONPATH=. python models/dnn/train.py --modalities t --out-dir results/meld_suite/dnn_t
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from dataset import SAMPLES_PKL

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keras.models import load_model

from dataset.preprocess.utils import constants as dataset_constants
from models.dnn.DNN import DNN
from models.keras_data import (
    load_preprocess_splits,
    samples_to_xy,
    save_run_meta,
)

logging.basicConfig(level=logging.INFO)


def _compute_test_metrics(y_true, y_pred, label_to_idx: dict) -> dict:
    import numpy as np
    from sklearn import metrics

    golds = np.asarray(y_true).astype(np.int64)
    pred_labels = np.asarray(y_pred).astype(np.int64)
    labels = np.arange(len(label_to_idx))
    names = list(label_to_idx.keys())
    p, r, f1, support = metrics.precision_recall_fscore_support(
        golds, pred_labels, labels=labels, zero_division=0
    )
    per_emotion = {}
    for i, name in enumerate(names):
        per_emotion[name] = {
            "f1": float(f1[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "support": int(support[i]),
        }
    acc = float(metrics.accuracy_score(golds, pred_labels))
    wf1 = float(metrics.f1_score(golds, pred_labels, average="weighted", zero_division=0))
    return {"accuracy": acc, "weighted_f1": wf1, "per_emotion": per_emotion}


def main():
    parser = argparse.ArgumentParser(description="DNN на MELD / samples.pkl")
    parser.add_argument("--modalities", choices=["a", "t", "at"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--samples-pkl", type=Path, default=None)
    parser.add_argument("--export-metrics-json", type=Path, default=None, metavar="PATH")
    args = parser.parse_args()

    mod = args.modalities
    pkl = args.samples_pkl or SAMPLES_PKL
    ckpt_dir = args.out_dir or (Path(__file__).resolve().parent / f"run_meld_{mod}")
    ckpt_dir = ckpt_dir.resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train, dev, test = load_preprocess_splits(pkl)
    X_train, y_train = samples_to_xy(train, mod)
    X_dev, y_dev = samples_to_xy(dev, mod)
    X_test, y_test = samples_to_xy(test, mod)

    feature_name = f"meld_{mod}"

    save_run_meta(
        ckpt_dir / "meld_meta.json",
        {
            "modalities": mod,
            "feature_dim": int(X_train.shape[1]),
            "feature_name": feature_name,
            "samples_pkl": str(pkl),
        },
    )

    dnn = DNN(epochs=args.epochs, checkpoint_dir=str(ckpt_dir))
    neuron_configs = [[128, 64], [256, 128]]
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-3]

    logging.info(
        "DNN mod=%s | train %s dim=%s",
        mod,
        len(y_train),
        X_train.shape[1],
    )

    _, best_y_test, best_y_pred = dnn.train_and_evaluate(
        feature_name,
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        neuron_configs,
        batch_sizes,
        learning_rates,
    )

    if best_y_test is None:
        logging.warning("Нет результатов")
        return

    model_path = ckpt_dir / f"best_dnn_{feature_name}.keras"
    test_loss = None
    if model_path.is_file():
        m = load_model(str(model_path))
        ev = m.evaluate(X_test, y_test, verbose=0)
        test_loss = float(ev[0])

    bundle = _compute_test_metrics(best_y_test, best_y_pred, dataset_constants.EMOTION_MAP)
    bundle["modalities"] = mod
    bundle["model"] = "dnn"
    bundle["keras_path"] = str(model_path.resolve()) if model_path.is_file() else None
    bundle["test_loss"] = test_loss

    if args.export_metrics_json:
        args.export_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_metrics_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Метрики: %s", args.export_metrics_json)

    logging.info(
        "Test accuracy %.4f weighted_f1 %.4f | %s",
        bundle["accuracy"],
        bundle["weighted_f1"],
        model_path,
    )


if __name__ == "__main__":
    main()
