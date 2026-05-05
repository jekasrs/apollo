"""
Тест сохранённой CNN на аудиопризнаках (отчёт как у Apollo eval).

Запуск из корня репозитория:
  python models/cnn/test.py
  python models/cnn/test.py --split dev
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from keras.models import load_model

from dataset.preprocess.utils import constants as dataset_constants
from models.keras_data import (
    load_preprocess_splits,
    load_run_meta,
    print_metrics_like_apollo,
    reshape_for_cnn,
    samples_to_xy,
)


def main():
    parser = argparse.ArgumentParser(description="CNN (audio): тест сохранённой модели")
    parser.add_argument("--pkl", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parent
    meta_path = ckpt_dir / "meld_meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"Нет {meta_path}. Сначала: python models/cnn/train.py"
        )
    meta = load_run_meta(meta_path)
    modalities = meta["modalities"]

    train, dev, test = load_preprocess_splits(args.pkl)
    samples = test if args.split == "test" else dev
    X, y = samples_to_xy(samples, modalities)
    X = reshape_for_cnn(X)

    model_path = args.model or (ckpt_dir / "best_cnn_meld.keras")
    if not model_path.is_file():
        raise SystemExit(f"Нет модели {model_path}")

    model = load_model(str(model_path))
    ev = model.evaluate(X, y, verbose=0)
    test_loss = float(ev[0])
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print_metrics_like_apollo(
        y, y_pred, dataset_constants.EMOTION_MAP, test_loss=test_loss
    )


if __name__ == "__main__":
    main()
