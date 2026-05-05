"""
Обучение CNN только на аудиопризнаках из ``samples.pkl`` (тот же пайплайн, что у Apollo).

Признаки: ``audio_features`` / ``mfcc`` → вектор размерности ``AUDIO_FEATURE_DIM`` (см. ``keras_data.utterance_feature_vector(..., "a")``).

Класс модели: ``CNN`` в файле ``CNN.py``.

Запуск из корня репозитория:
  python models/cnn/train.py
"""
import logging
import sys
from pathlib import Path

from dataset import SAMPLES_PKL

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keras.models import load_model

from dataset.preprocess.utils import constants as dataset_constants
from models.cnn.CNN import CNN
from models.keras_data import (
    load_preprocess_splits,
    print_metrics_like_apollo,
    reshape_for_cnn,
    samples_to_xy,
    save_run_meta,
)

# Только акустика (как в ``keras_data``: modalities ``"a"``).
AUDIO_ONLY = "a"

logging.basicConfig(level=logging.INFO)


def main():
    train, dev, test = load_preprocess_splits(SAMPLES_PKL)
    X_train, y_train = samples_to_xy(train, AUDIO_ONLY)
    X_dev, y_dev = samples_to_xy(dev, AUDIO_ONLY)
    X_test, y_test = samples_to_xy(test, AUDIO_ONLY)

    X_train = reshape_for_cnn(X_train)
    X_dev = reshape_for_cnn(X_dev)
    X_test = reshape_for_cnn(X_test)

    feature_name = "meld"
    ckpt_dir = Path(__file__).resolve().parent

    save_run_meta(
        ckpt_dir / "meld_meta.json",
        {
            "modalities": AUDIO_ONLY,
            "input_shape": list(X_train.shape[1:]),
            "feature_name": feature_name,
            "samples_pkl": str(SAMPLES_PKL),
        },
    )

    cnn = CNN(epochs=20, checkpoint_dir=str(ckpt_dir))
    cnn_configs = [[128, 256], [64, 128], [32, 64]]
    batch_sizes = [32]
    learning_rates = [1e-3]

    logging.info(
        "Train %s samples, dev %s, test %s; input shape %s (audio only)",
        len(y_train),
        len(y_dev),
        len(y_test),
        X_train.shape[1:],
    )

    _, best_y_test, best_y_pred = cnn.train_and_evaluate(
        feature_name,
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        cnn_configs,
        batch_sizes,
        learning_rates,
    )

    if best_y_test is None:
        logging.warning("Нет результатов")
        return

    model_path = ckpt_dir / f"best_cnn_{feature_name}.keras"
    test_loss = None
    if model_path.is_file():
        m = load_model(str(model_path))
        ev = m.evaluate(X_test, y_test, verbose=0)
        test_loss = float(ev[0])

    print_metrics_like_apollo(
        best_y_test,
        best_y_pred,
        dataset_constants.EMOTION_MAP,
        test_loss=test_loss,
    )


if __name__ == "__main__":
    main()
