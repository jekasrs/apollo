"""
Тест сохранённой DNN на dev/test из ``samples.pkl`` (отчёт как у Apollo eval).

Запуск из корня репозитория:
  python models/dnn/test.py
  python models/dnn/test.py --split dev
"""
import argparse
import sys
from pathlib import Path

from dataset import SAMPLES_PKL

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
    samples_to_xy,
)

def main():
    ckpt_dir = Path(__file__).resolve().parent
    meta_path = ckpt_dir / "meld_meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"Нет {meta_path}. Сначала обучите модель: python models/dnn/train.py"
        )
    meta = load_run_meta(meta_path)
    modalities = meta["modalities"]

    train, dev, test = load_preprocess_splits(SAMPLES_PKL)
    samples = test
    X, y = samples_to_xy(samples, modalities)

    model_path = "best_dnn_meld.keras"
    model = load_model(str(model_path))
    ev = model.evaluate(X, y, verbose=0)
    test_loss = float(ev[0])
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print_metrics_like_apollo(y, y_pred, dataset_constants.EMOTION_MAP, test_loss=test_loss)


if __name__ == "__main__":
    main()