"""
Обучение DNN на признаках из ``samples.pkl`` (тот же пайплайн, что у Apollo).

Запуск из корня репозитория:
  python models/dnn/train.py
"""
import logging
import sys
from pathlib import Path

from dataset import SAMPLES_PKL
from models.apollo.utils.constants import MODALITIES

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keras.models import load_model

from dataset.preprocess.utils import constants as dataset_constants
from models.dnn.DNN import DNN
from models.keras_data import (
    load_preprocess_splits,
    print_metrics_like_apollo,
    samples_to_xy,
    save_run_meta,
)

logging.basicConfig(level=logging.INFO)


def main():
    train, dev, test = load_preprocess_splits(SAMPLES_PKL)
    X_train, y_train = samples_to_xy(train, MODALITIES)
    X_dev, y_dev = samples_to_xy(dev, MODALITIES)
    X_test, y_test = samples_to_xy(test, MODALITIES)

    feature_name = "meld"
    ckpt_dir = Path(__file__).resolve().parent

    save_run_meta(
        ckpt_dir / "meld_meta.json",
        {
            "modalities": MODALITIES,
            "feature_dim": int(X_train.shape[1]),
            "feature_name": feature_name,
            "samples_pkl": str(SAMPLES_PKL),
        },
    )

    dnn = DNN(epochs=20, checkpoint_dir=str(ckpt_dir))
    neuron_configs = [[128, 64], [256, 128]]
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-3]

    logging.info(
        "Train %s samples, dev %s, test %s; feature dim %s",
        len(y_train),
        len(y_dev),
        len(y_test),
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
        logging.warning("Нет результатов (пустая сетка?)")
        return

    model_path = ckpt_dir / f"best_dnn_{feature_name}.keras"
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
