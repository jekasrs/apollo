"""
Добавляет к train MELD размеченные **английские тексты** из HuggingFace `dair-ai/emotion`
(6 классов твитов) с маппингом в 7 эмоций MELD.

Аудио: нет сырого сигнала — подставляется **средний** audio_features train MELD по тому же классу
(плюс слабый шум), чтобы размер 768 и канал at оставались валидными.

60–65% accuracy на MELD за один этот шаг нередко **недостижимы**; для роста обычно ещё нужны
дообученные Wav2Vec/MPNet на MELD (см. `dataset/finetune/`) и пересчёт `preprocess.py`.

Использование (из корня репо, сеть для первой загрузки датасета):
  pip install datasets
  PYTHONPATH=. python3 dataset/supplement/merge_hf_emotion6.py --out dataset/preprocess/samples/samples_meld_plus_emotion6.pkl
  PYTHONPATH=. python3 models/apollo/trainings/train.py --modalities at --use-pause --learning-rate 3e-4 --focal-gamma 1.5 \\
    --samples-pkl dataset/preprocess/samples/samples_meld_plus_emotion6.pkl
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import SAMPLES_PKL
from dataset.models.Sample import Sample
from dataset.preprocess.utils import utils as u
from dataset.preprocess.utils import constants as c
from dataset.preprocess.utils.hf_mpnet_encoder import make_text_embedder_for_preprocess

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# dair-ai/emotion: 0 sad, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise
# → MELD int label (EMOTION_MAP value)
_HF6_TO_MELD = {
    0: c.EMOTION_MAP["sadness"],
    1: c.EMOTION_MAP["joy"],
    2: c.EMOTION_MAP["joy"],  # love → joy
    3: c.EMOTION_MAP["anger"],
    4: c.EMOTION_MAP["fear"],
    5: c.EMOTION_MAP["surprise"],
}


def _per_class_mean_audio(train_samples: list) -> dict[int, np.ndarray]:
    acc: dict[int, list] = defaultdict(list)
    for s in train_samples:
        if getattr(s, "audio_features", None) is not None:
            v = np.asarray(s.audio_features, dtype=np.float64)
            acc[int(s.label)].append(v)
    out: dict[int, np.ndarray] = {}
    for k, arrs in acc.items():
        if arrs:
            out[k] = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-pkl",
        type=str,
        default=str(SAMPLES_PKL),
        help="Исходный MELD samples.pkl (train/dev/test)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="dataset/preprocess/samples/samples_meld_plus_emotion6.pkl",
        help="Куда записать train+доп, dev|test как в base",
    )
    p.add_argument("--max-samples", type=int, default=12_000, help="Сколько HF-строк максимум")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio-noise", type=float, default=0.02, help="std гаусса к mean-аудио")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    data = u.load_pickle(args.base_pkl)
    train: list = list(data["train"])
    dev = data["dev"]
    test = data["test"]
    s0 = train[0] if train else next(iter(test))
    pm = getattr(s0, "pause_norm_mu", None)
    ps = getattr(s0, "pause_norm_std", None)
    if pm is None or ps is None:
        pmv, psv = u.compute_pause_norm_stats(train)
        pm, ps = float(pmv), float(psv)
    else:
        pm, ps = float(pm), float(ps)
    n_classes = len(c.EMOTION_MAP)
    class_means = _per_class_mean_audio(train)
    global_mean = None
    if class_means:
        global_mean = np.mean(
            np.stack([class_means[k] for k in sorted(class_means.keys())], axis=0),
            axis=0,
        ).astype(np.float32)
    else:
        raise SystemExit("Нет train audio_features для mean")

    def audio_for_label(lab: int) -> list[float]:
        m = class_means.get(lab) if class_means else None
        if m is None:
            m = global_mean
        noise = rng.normal(0.0, args.audio_noise, size=m.shape).astype(np.float32)
        v = m + noise
        return v.reshape(-1).tolist()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    hf = load_dataset("emotion", split="train")
    n = min(len(hf), args.max_samples)
    rows = [hf[i] for i in range(n)]
    texts = [str(r["text"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    meld_labels = [_HF6_TO_MELD[int(l)] for l in labels]

    log.info("Кодируем %d текстов HF одним text encoder'ом, как в preprocess", len(texts))
    enc = make_text_embedder_for_preprocess()
    arr = enc.encode_batch(texts, batch_size=32)
    if arr.shape[0] != len(texts):
        raise SystemExit("encode_batch: размеры не сходятся")

    extra: list[Sample] = []
    for i, (t, mlab) in enumerate(tqdm(list(zip(texts, meld_labels)), desc="build Sample", total=len(texts))):
        e = np.asarray(arr[i], dtype=np.float32).reshape(-1)
        if e.shape[0] != 768:
            log.warning("dim %s != 768, skip row %d", e.shape, i)
            continue
        a = audio_for_label(mlab)
        if len(a) != 768:
            continue
        extra.append(
            Sample(
                utterance_id=0,
                text=t[:4000],
                audio_path="",
                label=mlab,
                dialogue_id=f"hf_emotion6_{i:08d}",
                start=0.0,
                end=1.0,
                embeddings=e.reshape(-1).tolist(),
                audio_features=a,
                speaker_name="hf_emotion6",
                pause=0.0,
                speaker_id=0,
                pause_norm_mu=pm,
                pause_norm_std=ps,
            )
        )

    new_train = train + extra
    out_data = {"train": new_train, "dev": dev, "test": test}
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    u.save_pickle(out_data, str(op))
    log.info("Готово: train %d + %d → %d | записано %s", len(train), len(extra), len(new_train), op)


if __name__ == "__main__":
    main()
