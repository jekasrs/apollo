import logging

import torch

from Dataset import SAMPLES_PATH
from Dataset.models.Apollo import Apollo
from Dataset.models.Coach import Coach
from Dataset.models.Dataset import Dataset
from Dataset.models.Optim import Optim
from Dataset.models.class_weights import compute_class_weights_from_samples
from Dataset.models.constants import (
    CLASS_WEIGHT_BETA,
    CLASSIFIER_HIDDEN_DIM,
    DEVICE,
    DIALOGUES_PER_BATCH,
    EPOCHS,
    FOCAL_GAMMA,
    LEARNING_RATE,
    MAX_GRAD_VALUE,
    MODALITY_PROJ_DIM,
    RUN_TEST_EACH_EPOCH,
    SMOKE_TEST,
    USE_FOCAL_LOSS,
    USE_INPUT_LAYER_NORM,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    MODALITIES,
)
from Dataset.utils.constants import EMOTION_MAP, DIMS
from Dataset.utils.io_utils import load_pickle
from Dataset.utils.pause_stats import compute_pause_norm_stats

logging.basicConfig(level=logging.INFO)


def main():
    data = load_pickle(f"Dataset/{SAMPLES_PATH}")
    logging.log(logging.INFO, f"Loaded data set MELD")

    train_samples = data["train"]
    dev_samples = data["dev"]
    test_samples = data["test"]

    if SMOKE_TEST:
        logging.info(
            "SMOKE_TEST mode: fewer epochs, smaller dialogues_per_batch "
            "(set SMOKE_TEST=False in Dataset/models/constants.py for full training)"
        )

    s0 = train_samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        pause_mu, pause_std = float(mu), float(std)
    else:
        pause_mu, pause_std = compute_pause_norm_stats(train_samples)
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        logging.warning(
            "В pickle нет pause_norm_mu/std — пересчитано по train. "
            "Обновите данные: python Dataset/preprocess.py"
        )
    ds_common = dict(
        dialogues_per_batch=DIALOGUES_PER_BATCH,
        modalities=MODALITIES,
        modality_feature_dim=DIMS[MODALITIES],
        pause_mu=pause_mu,
        pause_std=pause_std,
    )
    train_set = Dataset(train_samples, augment=True, **ds_common)
    dev_set = Dataset(dev_samples, augment=False, **ds_common)
    test_set = Dataset(test_samples, augment=False, **ds_common)

    logging.log(logging.INFO, f"Train batches: {len(train_set)}")
    logging.log(logging.INFO, f"Dev batches: {len(dev_set)}")
    logging.log(logging.INFO, f"Test batches: {len(test_set)}")

    class_weights = compute_class_weights_from_samples(
        train_samples, len(EMOTION_MAP), DEVICE, beta=CLASS_WEIGHT_BETA
    )
    logging.info("Class weights (balanced): %s", dict(zip(EMOTION_MAP.keys(), class_weights.tolist())))

    logging.log(logging.INFO, "Started creating Apollo model")
    model = Apollo(modalities=MODALITIES, device=DEVICE, class_weights=class_weights)
    model.to(DEVICE)

    opt = Optim(LEARNING_RATE, MAX_GRAD_VALUE, WEIGHT_DECAY)
    opt.set_parameters(params=model.parameters(), name="adamw")
    scheduler = opt.get_scheduler("reduceLR")

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

    logging.info("Starting training")
    best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s = coach.train()

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
            "use_input_layernorm": USE_INPUT_LAYER_NORM,
            "use_focal_loss": USE_FOCAL_LOSS,
            "focal_gamma": FOCAL_GAMMA,
            "label_smoothing": LABEL_SMOOTHING,
        },
        "dialogues_per_batch": DIALOGUES_PER_BATCH,
        "modality_feature_dim": DIMS[MODALITIES],
    }

    torch.save(checkpoint, "./checkpoints/model.pt")


if __name__ == "__main__":
    main()
