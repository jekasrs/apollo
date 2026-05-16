"""
Дообучение MPNet на Empathetic Dialogues (только текст), метки приведены к 7 классам MELD.

Датасет CSV: conv_id, utterance_idx, context, prompt, speaker_idx, utterance, ...
Колонка ``context`` — эмоция ситуации (32 типа); на каждую реплику в диалоге одна и та же метка.

Зачем: больше размеченного текста и близкая доменная лексика → лучше text encoder перед Apollo.

Дальше по цепочке (один из вариантов):
  1) Только ED: ``export APOLLO_FINETUNED_TEXT=<out_dir>`` → ``python dataset/preprocess/preprocess.py``
  2) ED затем уточнение на MELD: этот скрипт → ``finetune_text_meld.py --init_from <out_dir>`` → env и preprocess.

Пример:
  PYTHONPATH=. python3 dataset/finetune/finetune_text_empathetic.py \\
    --train_csv /path/train.csv --valid_csv /path/valid.csv \\
    --out_dir results/encoders/finetune_mpnet_ed_then_set_env
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
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

from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as preprocess_utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 32 ED context → имя класса MELD (ключ EMOTION_MAP)
ED_CONTEXT_TO_MELD: dict[str, str] = {
    "afraid": "fear",
    "angry": "anger",
    "annoyed": "anger",
    "anticipating": "surprise",
    "anxious": "fear",
    "apprehensive": "fear",
    "ashamed": "sadness",
    "caring": "neutral",
    "confident": "neutral",
    "content": "neutral",
    "devastated": "sadness",
    "disappointed": "sadness",
    "disgusted": "disgust",
    "embarrassed": "sadness",
    "excited": "joy",
    "faithful": "neutral",
    "furious": "anger",
    "grateful": "joy",
    "guilty": "sadness",
    "hopeful": "joy",
    "impressed": "surprise",
    "jealous": "anger",
    "joyful": "joy",
    "lonely": "sadness",
    "nostalgic": "sadness",
    "prepared": "neutral",
    "proud": "joy",
    "sad": "sadness",
    "sentimental": "sadness",
    "surprised": "surprise",
    "terrified": "fear",
    "trusting": "neutral",
}

NUM_LABELS = len(dataset_constants.EMOTION_MAP)
ID2LABEL = {i: name for name, i in dataset_constants.EMOTION_MAP.items()}
LABEL2ID = dataset_constants.EMOTION_MAP.copy()


def _read_ed_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _rows_to_xy(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    y: list[int] = []
    skipped = 0
    for _, row in df.iterrows():
        ctx = row.get("context")
        utt = row.get("utterance")
        if pd.isna(ctx) or pd.isna(utt):
            skipped += 1
            continue
        ctx_s = str(ctx).strip().lower()
        meld_name = ED_CONTEXT_TO_MELD.get(ctx_s)
        if meld_name is None:
            skipped += 1
            continue
        label = LABEL2ID[meld_name]
        t = preprocess_utils.clean_text(str(utt), remove_stopwords=False)
        if not t:
            skipped += 1
            continue
        texts.append(t)
        y.append(label)
    if skipped:
        log.info("Пропущено строк: %d (пусто / неизвестный context)", skipped)
    return texts, np.array(y, dtype=np.int64)


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
    p = argparse.ArgumentParser(description="MPNet на Empathetic Dialogues → метки MELD (7 классов)")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--valid_csv", type=Path, required=True)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/encoders/finetune_mpnet_empathetic"),
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--init_from",
        type=Path,
        default=None,
        help="Каталог HF-модели для старта (напр. чекпоинт после finetune_text_meld). Иначе microsoft/mpnet-base.",
    )
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_df = _read_ed_csv(args.train_csv)
    valid_df = _read_ed_csv(args.valid_csv)
    train_texts, train_y = _rows_to_xy(train_df)
    dev_texts, dev_y = _rows_to_xy(valid_df)
    if len(train_texts) < 100 or len(dev_texts) < 50:
        raise SystemExit("Слишком мало примеров после фильтрации — проверьте CSV и маппинг context.")

    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=train_y,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    log.info("Train %d  Dev %d  Class weights: %s", len(train_texts), len(dev_texts),
             {ID2LABEL[i]: float(cw[i]) for i in range(NUM_LABELS)})

    model_name = (
        str(args.init_from.resolve()) if args.init_from is not None else dataset_constants.HUGGINGFACE_MPNET_BASE
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MPNetForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    class _EdTextDataset(torch.utils.data.Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray, max_length: int) -> None:
            self._texts = texts
            self._labels = labels
            self._max_length = max_length

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

    train_ds = _EdTextDataset(train_texts, train_y, args.max_length)
    dev_ds = _EdTextDataset(dev_texts, dev_y, args.max_length)

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
        load_best_model_at_end=False,
        save_total_limit=1,
        logging_steps=200,
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
    log.info(
        "Готово. Для препроцесса Apollo: export APOLLO_FINETUNED_TEXT=%s\n"
        "Затем: PYTHONPATH=. python3 dataset/preprocess/preprocess.py",
        out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
