"""Coach управляет обучением, валидацией, тестированием."""
import copy
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.apollo.trainings.eval import evaluate_dataset
from models.apollo.utils.functions import batch_to_device

from models.apollo.utils.constants import (
    AUG_APPLY_PROB,
    AUG_AUDIO_STD,
    AUG_TEXT_STD,
    USE_TRAIN_AUGMENTATION,
)
from dataset.augment.augment import maybe_augment_input_tensor


class Coach:
    def __init__(
        self,
        train,
        dev,
        test,
        model,
        optimizer,
        scheduler,
        epochs,
        device,
        label_to_idx,
        run_test_each_epoch=True,
        early_stopping_patience: int = 0,
        dev_select_metric: str = "weighted_f1",
        checkpoint_save_dir: Path | str | None = None,
    ):
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
        self.early_stopping_patience = early_stopping_patience
        if dev_select_metric not in ("weighted_f1", "accuracy", "macro_f1"):
            raise ValueError(
                f"dev_select_metric must be weighted_f1|accuracy|macro_f1, got {dev_select_metric!r}"
            )
        self.dev_select_metric = dev_select_metric

        self.checkpoint_save_dir = (
            Path(checkpoint_save_dir) if checkpoint_save_dir is not None else Path("results/apollo_meld_at_r01")
        )

        # early stopping
        self.best_dev_score = None
        self.best_epoch = None
        self.best_state = None

    def train(self):
        best_dev_score, best_epoch, best_state = (
            self.best_dev_score,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        patience = self.early_stopping_patience
        epochs_no_improve = 0
        metric = self.dev_select_metric

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            dev_metrics = self.evaluate(test=False)
            dev_f1 = dev_metrics["weighted_f1"]
            dev_loss = dev_metrics["mean_loss"] * len(self.dev_set)
            dev_score = float(dev_metrics[metric])

            logging.info(
                "Epoch %d — train_loss: %.4f | dev: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f loss=%.4f",
                epoch,
                train_loss,
                dev_metrics["accuracy"],
                dev_metrics["macro_f1"],
                dev_metrics["weighted_f1"],
                dev_metrics["mean_loss"],
            )

            self.scheduler.step(dev_loss)
            if self.run_test_each_epoch:
                test_metrics = self.evaluate(test=True)
                test_f1 = np.array(list(test_metrics["per_class_f1"].values())).mean()
                test_f1s.append(test_f1)
            else:
                test_f1 = float("nan")

            improved = best_dev_score is None or dev_score > best_dev_score
            if improved:
                best_dev_score = dev_score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                self.checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": self.model},
                    str(self.checkpoint_save_dir / "best_dev_f1_model_.pt"),
                )
            else:
                if patience > 0:
                    epochs_no_improve += 1

            dev_f1s.append(dev_f1)
            train_losses.append(train_loss)

            if patience > 0 and epochs_no_improve >= patience:
                logging.info(
                    "Early stopping: dev %s не улучшался %d эпох подряд (patience=%d).",
                    metric,
                    patience,
                    patience,
                )
                break

        if not self.run_test_each_epoch:
            test_metrics = self.evaluate(test=True)
            test_f1 = np.array(list(test_metrics["per_class_f1"].values())).mean()
            test_f1s.append(test_f1)

        return best_dev_score, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()

        self.train_set.shuffle()
        for idx in tqdm(range(len(self.train_set)), desc="train epoch {}".format(epoch)):
            self.optimizer.zero_grad()
            data = self.train_set[idx]
            batch_to_device(data, self.device)

            if USE_TRAIN_AUGMENTATION and getattr(self.train_set, "augment", False):
                modalities = getattr(self.train_set, "modalities", "a")
                maybe_augment_input_tensor(
                    data["input_tensor"],
                    modalities,
                    AUG_APPLY_PROB,
                    AUG_AUDIO_STD,
                    AUG_TEXT_STD,
                    use_pause=getattr(self.train_set, "use_pause", True),
                )

            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.optimizer.step()

        return epoch_loss

    def evaluate(self, test=False):
        dataset = self.test_set if test else self.dev_set
        desc = "test" if test else "dev"
        return evaluate_dataset(
            self.model,
            dataset,
            self.device,
            self.label_to_idx,
            desc=desc,
            print_report=test,
        )