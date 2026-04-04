"""Coach управляет обучением, валидацией, тестированием."""
import copy

import numpy as np
import torch
from tqdm import tqdm

from models.apollo.utils.functions import batch_to_device
from trainings.eval import evaluate_dataset
from models.apollo.utils.constants import (
    AUG_APPLY_PROB,
    AUG_AUDIO_STD,
    AUG_TEXT_STD,
    MODALITIES,
    USE_TRAIN_AUGMENTATION,
)
from dataset.augment.augment import maybe_augment_input_tensor


class Coach:
    def __init__(self, train, dev, test, model, optimizer, scheduler, epochs, device, label_to_idx, run_test_each_epoch=True):
        self.train_set = train
        self.dev_set = dev
        self.test_set = test
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.label_to_idx = label_to_idx
        self.run_test_each_epoch = run_test_each_epoch

        # early stopping
        self.best_dev_f1 = None # лучший результат
        self.best_epoch = None  # на какой эпохе
        self.best_state = None  # веса модели

    def train(self):
        # Early stopping
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []

        # Train
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch) # 1 - тренировка
            dev_f1, dev_loss = self.evaluate()   # 2 - валидация

            self.scheduler.step(dev_loss) # 3 - регуляризация lr
            if self.run_test_each_epoch:
                test_f1, _ = self.evaluate(test=True) # 4 - тестирование
                test_f1 = np.array(list(test_f1.values())).mean() # 5 - усреднение f1, так как f1 отдельно по каждому классу
                test_f1s.append(test_f1)
            else:
                test_f1 = float("nan")

            if best_dev_f1 is None or dev_f1 > best_dev_f1: # 6 - если модель стала лучше, то запоминаем веса
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save( # 7 - сохраняем
                    {"state_dict": self.model},
                    "trainings/checkpoints/best_dev_f1_model_.pt"
                )

            dev_f1s.append(dev_f1)
            train_losses.append(train_loss)

        if not self.run_test_each_epoch:
            test_f1, _ = self.evaluate(test=True)
            test_f1 = np.array(list(test_f1.values())).mean()
            test_f1s.append(test_f1)

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()

        self.train_set.shuffle()
        for idx in tqdm(range(len(self.train_set)), desc="train epoch {}".format(epoch)):
            self.optimizer.zero_grad()
            data = self.train_set[idx]
            batch_to_device(data, self.device)

            if USE_TRAIN_AUGMENTATION and getattr(self.train_set, "augment", False):
                maybe_augment_input_tensor(
                    data["input_tensor"],
                    MODALITIES,
                    AUG_APPLY_PROB,
                    AUG_AUDIO_STD,
                    AUG_TEXT_STD,
                )

            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.optimizer.step()

        return epoch_loss

    def evaluate(self, test=False):
        dataset = self.test_set if test else self.dev_set
        desc = "test" if test else "dev"
        out = evaluate_dataset(
            self.model,
            dataset,
            self.device,
            self.label_to_idx,
            desc=desc,
            print_report=test,
        )
        if test:
            return out["per_class_f1"], out["mean_loss"] * len(dataset)
        return out["weighted_f1"], out["mean_loss"] * len(dataset)