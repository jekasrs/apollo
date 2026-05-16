"""
Дообучение MPNet (microsoft/mpnet-base) на классификацию эмоций MELD по тексту реплики.

Использует те же train/dev, что в ``samples.pkl`` (нужен предварительный запуск preprocess).

Сохранение: ``--out_dir`` (Trainer), далее препроцесс:
  export APOLLO_FINETUNED_TEXT=/path/to/out_dir
  python Dataset/preprocess/preprocess.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    MPNetForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from dataset import SAMPLES_PKL
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as preprocess_utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

NUM_LABELS = len(dataset_constants.EMOTION_MAP)
ID2LABEL = {i: name for name, i in dataset_constants.EMOTION_MAP.items()}
LABEL2ID = dataset_constants.EMOTION_MAP.copy()


def _label_to_int(y) -> int:
    if hasattr(y, "item"):
        return int(y.item())
    return int(y)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights.float()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self._class_weights.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/encoders/finetune_mpnet_meld"),
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--init_from",
        type=Path,
        default=None,
        help="Каталог HF-модели (напр. после finetune_text_empathetic.py); иначе mpnet-base.",
    )
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = preprocess_utils.load_pickle(SAMPLES_PKL)
    train = data["train"]
    dev = data["dev"]
    if not train or not dev:
        raise SystemExit("train/dev в samples.pkl пусты. Сначала: python Dataset/preprocess/preprocess.py")

    train_texts = [s.text for s in train]
    dev_texts = [s.text for s in dev]
    train_y = np.array([_label_to_int(s.label) for s in train], dtype=np.int64)
    dev_y = np.array([_label_to_int(s.label) for s in dev], dtype=np.int64)

    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=train_y,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    log.info("Class weights: %s", {ID2LABEL[i]: float(cw[i]) for i in range(NUM_LABELS)})

    model_name = (
        str(args.init_from.resolve()) if args.init_from is not None else dataset_constants.HUGGINGFACE_MPNET_BASE
    )
    if args.init_from is not None:
        log.info("Старт с весов: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MPNetForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    class _MeldTextDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            texts: list[str],
            labels: np.ndarray,
            max_length: int | None = None,
        ) -> None:
            self._texts = texts
            self._labels = labels
            self._max_length = (
                int(max_length) if max_length is not None else dataset_constants.TEXT_ENCODER_MAX_LENGTH
            )

        def __len__(self) -> int:
            return len(self._texts)

        def __getitem__(self, i: int) -> dict:
            enc = tokenizer(
                self._texts[i],
                truncation=True,
                max_length=self._max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self._labels[i], dtype=torch.long),
            }

    train_ds = _MeldTextDataset(train_texts, train_y)
    dev_ds = _MeldTextDataset(dev_texts, dev_y)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pr = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, pr)),
            "f1_weighted": float(f1_score(labels, pr, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(labels, pr, average="macro", zero_division=0)),
        }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Иначе при несовпадении имён весов в чекпоинтах (LayerNorm) HF может сбросить лучшую сеть
        load_best_model_at_end=False,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("Готово. Укажите путь: export APOLLO_FINETUNED_TEXT=%s", out_dir.resolve())


if __name__ == "__main__":
    main()
