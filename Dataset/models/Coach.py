"""
Coach управляет обучением, валидацией, тестированием.
"""
import copy
import time

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from Dataset.utils.constants import EMOTION_MAP


class Coach:
    def __init__(self, train, dev, test, model, opt, sched, epochs, device, label_to_idx):
        self.experiment = None
        self.train_set = train
        self.dev_set = dev
        self.test_set = test
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.epochs = epochs
        self.device = device
        self.label_to_idx = label_to_idx
        self.label_dict = EMOTION_MAP

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
            test_f1, _ = self.evaluate(test=True) # 4 - тестирование
            test_f1 = np.array(list(test_f1.values())).mean() # 5 - усреднение f1, так как f1 отдельно по каждому классу

            if best_dev_f1 is None or dev_f1 > best_dev_f1: # 6 - если модель стала лучше, то запоминаем веса
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save( # 7 - сохраняем
                    {"state_dict": self.model},
                    "checkpoints/best_dev_f1_model_.pt"
                )

            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        self.train_set.shuffle()
        for idx in tqdm(range(len(self.train_set)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.train_set[idx]
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.device)

            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        return epoch_loss

    def evaluate(self, test=False):
        dev_loss = 0
        dataset = self.test_set if test else self.dev_set
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))
                nll = self.model.get_loss(data)
                dev_loss += nll.item()

            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4
                    )
                )

                happy = metrics.f1_score(
                    golds[:, 0], preds[:, 0], average="weighted"
                )
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(
                    golds[:, 2], preds[:, 2], average="weighted"
                )
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(
                    golds[:, 4], preds[:, 4], average="weighted"
                )
                fear = metrics.f1_score(
                    golds[:, 5], preds[:, 5], average="weighted"
                )

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }

                self.experiment.log_metric(
                    "accuracy score", metrics.accuracy_score(golds, preds)
                )
                self.experiment.log_metric("happiness_f1", happy)
                self.experiment.log_metric("sadness_f1", sad)
                self.experiment.log_metric("anger_f1", anger)
                self.experiment.log_metric("surprise_f1", surprise)
                self.experiment.log_metric("disgust_f1", disgust)
                self.experiment.log_metric("fear_f1", fear)

        return f1, dev_loss