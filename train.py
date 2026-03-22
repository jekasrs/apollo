import logging

import torch

from Dataset import SAMPLES_PATH
from Dataset.models.Apollo import Apollo
from Dataset.models.Coach import Coach
from Dataset.models.Dataset import Dataset
from Dataset.models.Optim import Optim
from Dataset.models.constants import DEVICE, LEARNING_RATE, MAX_GRAD_VALUE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE
from Dataset.utils.constants import EMOTION_MAP, DIMS
from Dataset.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO)


def main():
    # Загрузка данных
    data = load_pickle(f"Dataset/{SAMPLES_PATH}")
    logging.log(logging.INFO, f"Loaded data set MELD")

    train_set = Dataset(data["train"], batch_size=BATCH_SIZE, modalities="at", dataset_embedding_dims=DIMS["at"])
    dev_set = Dataset(data["dev"], batch_size=BATCH_SIZE, modalities="at", dataset_embedding_dims=DIMS["at"])
    test_set = Dataset(data["test"], batch_size=BATCH_SIZE, modalities="at", dataset_embedding_dims=DIMS["at"])

    logging.log(logging.INFO, f"A train array len={len(train_set)}")
    logging.log(logging.INFO, f"A dev array len={len(dev_set)}")
    logging.log(logging.INFO, f"A test array len={len(test_set)}")

    # Создаем модель
    logging.log(logging.INFO, "Started creating Apollo model")
    model = Apollo( modalities="at", device=DEVICE)
    model.to(DEVICE)

    # Создаем оптимизатор, получаем scheduler
    opt = Optim(LEARNING_RATE, MAX_GRAD_VALUE, WEIGHT_DECAY)
    opt.set_parameters(params=model.parameters(), name="adam")
    scheduler = opt.get_scheduler("reduceLR")

    # Создаем тренера для обучения и валидации
    coach = Coach(train_set, dev_set, test_set, model, opt, scheduler, EPOCHS, DEVICE, EMOTION_MAP)

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
        "model_args": None
    }

    model_file = "./checkpoints/model.pt"
    torch.save(checkpoint, model_file)

if __name__ == '__main__':
    main()