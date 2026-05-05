"""
Обучение LSTM на признаках из ``samples.pkl`` (тот же пайплайн, что у Apollo).

Класс модели: ``LSTMM`` в файле ``LSTMM.py``.

Запуск из корня репозитория:
  python models/lstm/train.py
"""
import argparse
import logging
import sys
from pathlib import Path

from dataset import SAMPLES_PKL

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keras.models import load_model

from dataset.preprocess.utils import constants as dataset_constants
from models.apollo.utils import constants as arguments_and_constants
from models.keras_data import (
    load_preprocess_splits,
    print_metrics_like_apollo,
    reshape_for_lstm,
    samples_to_xy,
    save_run_meta,
)
from models.lstm.LSTMM import LSTMM

logging.basicConfig(level=logging.INFO)


def main():
    train, dev, test = load_preprocess_splits(SAMPLES_PKL)
    X_train, y_train = samples_to_xy(train, arguments_and_constants.MODALITIES)
    X_dev, y_dev = samples_to_xy(dev, arguments_and_constants.MODALITIES)
    X_test, y_test = samples_to_xy(test, arguments_and_constants.MODALITIES)

    X_train = reshape_for_lstm(X_train)
    X_dev = reshape_for_lstm(X_dev)
    X_test = reshape_for_lstm(X_test)

    feature_name = "meld"
    ckpt_dir = Path(__file__).resolve().parent

    save_run_meta(
        ckpt_dir / "meld_meta.json",
        {
            "modalities": arguments_and_constants.MODALITIES,
            "input_shape": list(X_train.shape[1:]),
            "feature_name": feature_name,
            "samples_pkl": str(SAMPLES_PKL),
        },
    )

    lstm = LSTMM(epochs=20, checkpoint_dir=str(ckpt_dir))
    lstm_configs = [[128, 64], [256, 128]]
    batch_sizes = [32]
    learning_rates = [1e-3]

    logging.info(
        "Train %s samples, dev %s, test %s; input shape %s",
        len(y_train),
        len(y_dev),
        len(y_test),
        X_train.shape[1:],
    )

    _, best_y_test, best_y_pred = lstm.train_and_evaluate(
        feature_name,
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        lstm_configs,
        batch_sizes,
        learning_rates,
    )

    if best_y_test is None:
        logging.warning("Нет результатов")
        return

    model_path = ckpt_dir / f"best_lstm_{feature_name}.keras"
    test_loss = None
    if model_path.is_file():
        m = load_model(str(model_path))
        ev = m.evaluate(X_test, y_test, verbose=0)
        test_loss = float(ev[0])

    print_metrics_like_apollo(
        best_y_test, best_y_pred, dataset_constants.EMOTION_MAP, test_loss=test_loss
    )


if __name__ == "__main__":
    main()
