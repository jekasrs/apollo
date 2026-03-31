import logging

import torch

from Dataset import SAMPLES_PATH
from Dataset.models.Apollo import Apollo
from Dataset.models.Coach import Coach
from Dataset.models.Dataset import Dataset
from Dataset.models.Optim import Optim
from Dataset.models.class_weights import compute_class_weights_from_samples
from Dataset.models.constants import (
    BATCH_SIZE,
    CLASSIFIER_HIDDEN_DIM,
    DEVICE,
    DEV_MAX_SAMPLES,
    EPOCHS,
    FOCAL_GAMMA,
    LEARNING_RATE,
    MAX_GRAD_VALUE,
    MODALITY_PROJ_DIM,
    RUN_TEST_EACH_EPOCH,
    SMOKE_TEST,
    TEST_MAX_SAMPLES,
    TRAIN_MAX_SAMPLES,
    USE_FOCAL_LOSS,
    USE_INPUT_LAYERNORM,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    MODALITIES,
)
from Dataset.utils.constants import EMOTION_MAP, DIMS
from Dataset.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO)


def _limit(xs, n):
    return xs if n is None else xs[:n]


def main():
    # Загрузка данных
    data = load_pickle(f"Dataset/{SAMPLES_PATH}")
    logging.log(logging.INFO, f"Loaded data set MELD")

    train_samples = _limit(data["train"], TRAIN_MAX_SAMPLES)
    dev_samples = _limit(data["dev"], DEV_MAX_SAMPLES)
    test_samples = _limit(data["test"], TEST_MAX_SAMPLES)

    if SMOKE_TEST:
        logging.info(
            "SMOKE_TEST mode: smaller data, fewer epochs, bucketed GNN relations "
            "(set SMOKE_TEST=False in Dataset/models/constants.py for full training)"
        )

    train_set = Dataset(train_samples, batch_size=BATCH_SIZE, modalities=MODALITIES, dataset_embedding_dims=DIMS[MODALITIES])
    dev_set = Dataset(dev_samples, batch_size=BATCH_SIZE, modalities=MODALITIES, dataset_embedding_dims=DIMS[MODALITIES])
    test_set = Dataset(test_samples, batch_size=BATCH_SIZE, modalities=MODALITIES, dataset_embedding_dims=DIMS[MODALITIES])

    logging.log(logging.INFO, f"A train array len={len(train_set)}")
    logging.log(logging.INFO, f"A dev array len={len(dev_set)}")
    logging.log(logging.INFO, f"A test array len={len(test_set)}")

    class_weights = compute_class_weights_from_samples(train_samples, len(EMOTION_MAP), DEVICE)
    logging.info("Class weights (balanced): %s", dict(zip(EMOTION_MAP.keys(), class_weights.tolist())))

    # Создаем модель
    logging.log(logging.INFO, "Started creating Apollo model")
    model = Apollo(modalities=MODALITIES, device=DEVICE, class_weights=class_weights)
    model.to(DEVICE)

    # Создаем оптимизатор, получаем scheduler
    opt = Optim(LEARNING_RATE, MAX_GRAD_VALUE, WEIGHT_DECAY)
    opt.set_parameters(params=model.parameters(), name="adam")
    scheduler = opt.get_scheduler("reduceLR")

    # Создаем тренера для обучения и валидации
    coach = Coach(
        train=train_set,
        dev=dev_set,
        test=test_set,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        label_to_idx=EMOTION_MAP,
        run_test_each_epoch=RUN_TEST_EACH_EPOCH,
    )

    logging.log(logging.INFO, "Apollo model created successfully")

    # Тренируем
    logging.info("Starting training")
    best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s = coach.train()

    # Сохраняем лучшую модель
    checkpoint = {
        "best_dev_f1": best_dev_f1,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "train_losses": train_losses,
        "dev_f1s": dev_f1s,
        "test_f1s": test_f1s,
        "class_weights": class_weights.detach().cpu(),
        "model_args": {
            "modalities": MODALITIES,
            "modality_proj_dim": MODALITY_PROJ_DIM,
            "classifier_hidden_dim": CLASSIFIER_HIDDEN_DIM,
            "use_input_layernorm": USE_INPUT_LAYERNORM,
            "use_focal_loss": USE_FOCAL_LOSS,
            "focal_gamma": FOCAL_GAMMA,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }

    model_file = "./checkpoints/model.pt"
    torch.save(checkpoint, model_file)

if __name__ == '__main__':
    main()