"""
Дообучение Wav2Vec2 (facebook/wav2vec2-base) на 7 эмоций MELD по сегменту wav.

Train/dev — из ``samples.pkl`` (тот же split, что и Apollo).

После: ``export APOLLO_FINETUNED_WAV2VEC=/path/to/out`` и пересборка ``preprocess.py``.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, get_linear_schedule_with_warmup

from dataset import SAMPLES_PKL
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as preprocess_utils
from dataset.preprocess.utils.constants import EMOTION_MAP, SAMPLE_RATE, WAV2VEC_MODEL_NAME

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

NUM_LABELS = len(EMOTION_MAP)
ID2LABEL = {i: n for n, i in EMOTION_MAP.items()}


def _label_to_int(y) -> int:
    if hasattr(y, "item"):
        return int(y.item())
    return int(y)


class _AudioDS(Dataset):
    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> tuple:
        s = self._samples[i]
        path = Path(s.audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Нет файла: {path}")
        y, _ = preprocess_utils.load_audio_segment(str(path), sr=SAMPLE_RATE)
        y = preprocess_utils.normalize_audio(y)
        lab = _label_to_int(s.label)
        return y.astype(np.float32), lab


def _collate(batch, feature_extractor: Wav2Vec2FeatureExtractor, device: torch.device):
    waves = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    fe = feature_extractor(
        [w.tolist() for w in waves],
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_values": fe["input_values"].to(device),
        "attention_mask": fe.get("attention_mask"),
        "labels": labels,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/encoders/finetune_wav2vec_meld"),
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Случайная подвыборка train (ускорение; None = весь train)",
    )
    args = p.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch_size >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = preprocess_utils.load_pickle(SAMPLES_PKL)
    train = data["train"]
    dev = data["dev"]
    if not train or not dev:
        raise SystemExit("train/dev пусты. Сначала: python Dataset/preprocess/preprocess.py")

    if args.max_train_samples and args.max_train_samples < len(train):
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(train))[: int(args.max_train_samples)]
        train = [train[i] for i in idx]
        log.info("Train ограничен: %d реплик (из %d)", len(train), len(data["train"]))

    train_y = np.array([_label_to_int(s.label) for s in train])
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=train_y,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)
    log.info("Class weights: %s", {ID2LABEL[i]: float(cw[i]) for i in range(NUM_LABELS)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        WAV2VEC_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=EMOTION_MAP,
    )
    model.to(device)

    train_ds = _AudioDS(train)
    dev_ds = _AudioDS(dev)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: b,
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: b)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for raw_batch in pbar:
            opt.zero_grad()
            batch = _collate(raw_batch, fe, device)
            att = batch.get("attention_mask")
            if att is not None:
                att = att.to(device)
            out = model(
                input_values=batch["input_values"],
                attention_mask=att,
                labels=None,
            )
            logits = out.logits
            w = class_weights.to(logits.device)
            loss = F.cross_entropy(logits, batch["labels"], weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for raw_batch in dev_loader:
                batch = _collate(raw_batch, fe, device)
                att = batch.get("attention_mask")
                if att is not None:
                    att = att.to(device)
                logits = model(
                    input_values=batch["input_values"],
                    attention_mask=att,
                ).logits
                pr = logits.argmax(-1).cpu().numpy()
                y_pred.extend(pr.tolist())
                y_true.extend(batch["labels"].cpu().numpy().tolist())
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        log.info("dev acc=%.4f f1_w=%.4f f1_macro=%.4f", acc, f1w, f1m)
        if f1w > best_f1:
            best_f1 = f1w
            model.save_pretrained(out_dir)
            fe.save_pretrained(out_dir)
            log.info("  → сохранено в %s (best f1_w=%.4f)", out_dir, best_f1)

    if best_f1 < 0:
        model.save_pretrained(out_dir)
        fe.save_pretrained(out_dir)
    log.info("Готово. export APOLLO_FINETUNED_WAV2VEC=%s", out_dir.resolve())


if __name__ == "__main__":
    main()
