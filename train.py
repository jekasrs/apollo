import logging

import torch

from Dataset.models.Apollo import Apollo
from Dataset.models.Coach import Coach
from Dataset.utils.constants import SAMPLES_PATH
from Dataset.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Загрузка данных
    data = load_pickle(f"Dataset/{SAMPLES_PATH}")
    logging.log(logging.INFO, f"Loaded data set MELD")

    train = data.get('train')
    dev = data.get('dev')
    test = data.get('train')
    logging.log(logging.INFO, f"Train array len={len(train)}, dev array len={len(dev)}, test array len={len(test)}")

    # Создаем модель
    logging.log(logging.INFO, f"Started creating Apollo model")
    model = Apollo()
    # opt = Optim(learning_rate, max_grad_value, weight_decay)
    # opt.set_parameters(model.parameters())
    coach = Coach(train, dev, test, model)
    logging.log(logging.INFO, f"Finished creating Apollo model")

    # Тренируем
    logging.log(logging.INFO, f"Start training")
    ret = coach.train()

    # Сохарняем
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }

    model_file = "./model_checkpoints/model.pt"
    torch.save(checkpoint, model_file)