import logging

import torch

from dataset import SAMPLES_PKL
from dataset.models.Dataset import Dataset
from dataset.preprocess.utils import utils as dataset_utils
from dataset.preprocess.utils import constants as dataset_constants
from models.apollo.Apollo import Apollo
from models.apollo.Coach import Coach
from models.apollo.Optim import Optim
from models.apollo.utils import constants as arguments_and_constants
from models.apollo.utils.class_weights import compute_class_weights_from_samples

logging.basicConfig(level=logging.INFO)


def main():
    data = dataset_utils.load_pickle(f"{SAMPLES_PKL}")
    # /Users/evsmirnovalek/PycharmProjects/apollo/dataset/preprocess/samples/samples.pkl
    logging.log(logging.INFO, f"Loaded data set MELD")

    train_samples = data["train"]
    dev_samples = data["dev"]
    test_samples = data["test"]

    if arguments_and_constants.SMOKE_TEST:
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
        pause_mu, pause_std = dataset_utils.compute_pause_norm_stats(train_samples)
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        logging.warning(
            "В pickle нет pause_norm_mu/std — пересчитано по train. "
            "Обновите данные: python Dataset/preprocess.py"
        )
    ds_common = dict(
        dialogues_per_batch=arguments_and_constants.DIALOGUES_PER_BATCH,
        modalities=arguments_and_constants.MODALITIES,
        modality_feature_dim=arguments_and_constants.DIMS[arguments_and_constants.MODALITIES],
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
        train_samples, len(dataset_constants.EMOTION_MAP), arguments_and_constants.DEVICE, beta=arguments_and_constants.CLASS_WEIGHT_BETA
    )
    logging.info("Class weights (balanced): %s", dict(zip(dataset_constants.EMOTION_MAP.keys(), class_weights.tolist())))

    logging.log(logging.INFO, "Started creating Apollo model")
    model = Apollo(modalities=arguments_and_constants.MODALITIES, device=arguments_and_constants.DEVICE, class_weights=class_weights)
    model.to(arguments_and_constants.DEVICE)

    opt = Optim(arguments_and_constants.LEARNING_RATE, arguments_and_constants.MAX_GRAD_VALUE, arguments_and_constants.WEIGHT_DECAY)
    opt.set_parameters(params=model.parameters(), name="adamw")
    scheduler = opt.get_scheduler("reduceLR")

    coach = Coach(
        train=train_set,
        dev=dev_set,
        test=test_set,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        epochs=arguments_and_constants.EPOCHS,
        device=arguments_and_constants.DEVICE,
        label_to_idx=dataset_constants.EMOTION_MAP,
        run_test_each_epoch=arguments_and_constants.RUN_TEST_EACH_EPOCH,
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
            "modalities": arguments_and_constants.MODALITIES,
            "modality_proj_dim": arguments_and_constants.MODALITY_PROJ_DIM,
            "classifier_hidden_dim": arguments_and_constants.CLASSIFIER_HIDDEN_DIM,
            "use_input_layernorm": arguments_and_constants.USE_INPUT_LAYER_NORM,
            "use_focal_loss": arguments_and_constants.USE_FOCAL_LOSS,
            "focal_gamma": arguments_and_constants.FOCAL_GAMMA,
            "label_smoothing": arguments_and_constants.LABEL_SMOOTHING,
        },
        "dialogues_per_batch": arguments_and_constants.DIALOGUES_PER_BATCH,
        "modality_feature_dim": arguments_and_constants.DIMS[arguments_and_constants.MODALITIES],
    }

    torch.save(checkpoint, "trainings/checkpoints/model.pt")


if __name__ == "__main__":
    main()
