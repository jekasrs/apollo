"""
Классический пайплайн признаков для сравнения бейзлайнов (Keras DNN/LSTM/CNN на ``samples.pkl``):

- **Текст:** Word2Vec или FastText (обучение только на **train** после сплита по диалогам,
  вектор реплики — среднее по найденным словам после ``clean_text`` из ``utils``).
- **Аудио:** усреднённые по времени **MFCC** или **Mel-спектрограмма** (log-power, dB).

Сплит, паузы и нормализация паузы — как в основном ``preprocess.py``.

Запуск из корня репозитория::

  PYTHONPATH=. python dataset/preprocess/preprocess_classical.py \\
    --text word2vec --audio mfcc --out dataset/preprocess/samples/samples_w2v_mfcc.pkl
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
from gensim.models import FastText as GensimFastText
from gensim.models import Word2Vec
from tqdm import tqdm

from dataset import AUDIO_DIR, DATASET_PATH
from dataset.models.Sample import Sample
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as preprocess_utils

log = logging.getLogger(__name__)


@dataclass
class _Pending:
    utterance_id: int
    dialogue_id: Any
    text_raw: str
    text_clean: str
    audio_path: Path
    label: int
    speaker: str
    start: float
    end: float
    audio: np.ndarray


def _tokenize(s: str) -> List[str]:
    return [t for t in str(s).split() if t]


def _sentence_vec(kv, tokens: List[str], dim: int) -> np.ndarray:
    if not tokens:
        return np.zeros(dim, dtype=np.float32)
    vecs = []
    for w in tokens:
        try:
            vecs.append(kv.get_vector(w, norm=False))
        except KeyError:
            pass
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def _audio_mfcc_mean(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    import librosa

    if y.size == 0:
        return np.zeros(n_mfcc, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    v = np.mean(mfcc, axis=1).astype(np.float32)
    return v


def _audio_mel_mean_db(y: np.ndarray, sr: int, n_mels: int = 64, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    import librosa

    if y.size == 0:
        return np.zeros(n_mels, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.mean(mel_db, axis=1).astype(np.float32)


def _fit_text_backend(name: str, sentences_tokenized: List[List[str]], dim: int, epochs: int):
    name_l = name.lower().strip()
    common = dict(
        vector_size=dim,
        window=5,
        min_count=1,
        workers=1,
        epochs=epochs,
        seed=dataset_constants.RANDOM_STATE,
    )
    if name_l == "word2vec":
        return Word2Vec(sentences=sentences_tokenized, **common)
    if name_l == "fasttext":
        return GensimFastText(sentences=sentences_tokenized, **common)
    raise ValueError(f"Unknown text backend: {name!r} (use word2vec or fasttext)")


def build_classical_meld(
    *,
    text_backend: str,
    audio_backend: str,
    text_dim: int = 100,
    text_train_epochs: int = 15,
    n_mfcc: int = 40,
    n_mels: int = 64,
) -> tuple[list, list, list]:
    df = preprocess_utils.load_dataset(DATASET_PATH, AUDIO_DIR)
    pending: list[_Pending] = []
    prev_did = None
    utt_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MELD classical (load audio)"):
        did = row["dialogue_id"]
        if prev_did != did:
            utt_idx = 0
        prev_did = did
        text_raw = preprocess_utils.text_for_neural_encoder(row["utterance"])
        text_clean = preprocess_utils.clean_text(row["utterance"], remove_stopwords=True)
        y, sr = preprocess_utils.load_audio_segment(row["path_to_audio"])
        y = preprocess_utils.normalize_audio(y)
        pending.append(
            _Pending(
                utterance_id=utt_idx,
                dialogue_id=did,
                text_raw=text_raw,
                text_clean=text_clean,
                audio_path=Path(row["path_to_audio"]),
                label=int(row["emotion"]),
                speaker=str(row["speaker"]),
                start=float(row["start"]),
                end=float(row["end"]),
                audio=y.astype(np.float32),
            )
        )
        utt_idx += 1

    shells: list[Any] = []
    for p in pending:
        shells.append(
            Sample(
                utterance_id=p.utterance_id,
                text=p.text_raw,
                audio_path=str(p.audio_path),
                label=p.label,
                dialogue_id=p.dialogue_id,
                start=p.start,
                end=p.end,
                embeddings=np.zeros(text_dim, dtype=np.float32),
                audio_features=np.zeros(n_mfcc if audio_backend == "mfcc" else n_mels, dtype=np.float32),
                speaker_name=p.speaker,
            )
        )

    preprocess_utils.assign_pause_until_next_in_dialogues(shells)
    train, dev, test = preprocess_utils.split_samples_by_dialogue(
        shells,
        test_size=dataset_constants.TEST_SIZE,
        dev_size=dataset_constants.DEV_SIZE,
        random_state=dataset_constants.RANDOM_STATE,
    )

    by_key_text = {(p.dialogue_id, p.utterance_id): p.text_clean for p in pending}
    audio_by_key = {(p.dialogue_id, p.utterance_id): p.audio for p in pending}

    def clean_for_shell(s):
        return by_key_text[(s.dialogue_id, s.utterance_id)]

    trains_sent = [_tokenize(clean_for_shell(s)) for s in train]

    log.info(
        "Fitting %s on %d train utterances (%d tokenized sentences, dim=%d)",
        text_backend,
        len(trains_sent),
        sum(1 for t in trains_sent if t),
        text_dim,
    )
    wmodel = _fit_text_backend(text_backend, trains_sent, dim=text_dim, epochs=text_train_epochs)
    kv = wmodel.wv

    pause_mu, pause_std = preprocess_utils.compute_pause_norm_stats(train)
    pm, ps = float(pause_mu), float(pause_std)

    def fill_sample(shell: Sample) -> Sample:
        tclean = clean_for_shell(shell)
        toks = _tokenize(tclean)
        emb = _sentence_vec(kv, toks, text_dim)

        key = (shell.dialogue_id, shell.utterance_id)
        y = audio_by_key.get(key)
        if y is None:
            y = np.zeros(1, dtype=np.float32)

        sr = dataset_constants.SAMPLE_RATE
        if audio_backend.lower() == "mfcc":
            au = _audio_mfcc_mean(y, sr, n_mfcc=n_mfcc)
        elif audio_backend.lower() in {"mel", "melspectrogram", "spectrogram"}:
            au = _audio_mel_mean_db(y, sr, n_mels=n_mels)
        else:
            raise ValueError(audio_backend)

        emb = preprocess_utils.l2_normalize_rows(emb[np.newaxis, :])[0]
        au = preprocess_utils.l2_normalize_rows(au[np.newaxis, :])[0]

        return Sample(
            utterance_id=shell.utterance_id,
            text=shell.text,
            audio_path=shell.audio_path,
            label=shell.label,
            dialogue_id=shell.dialogue_id,
            start=shell.start,
            end=shell.end,
            embeddings=emb.astype(np.float32),
            audio_features=au.astype(np.float32),
            speaker_name=shell.speaker_name,
            pause=shell.pause,
            pause_norm_mu=pm,
            pause_norm_std=ps,
        )

    log.info("Vectorizing splits (audio=%s)…", audio_backend)
    train_f = [fill_sample(s) for s in tqdm(train, desc="train vectors")]
    dev_f = [fill_sample(s) for s in tqdm(dev, desc="dev vectors")]
    test_f = [fill_sample(s) for s in tqdm(test, desc="test vectors")]
    return train_f, dev_f, test_f


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Классические признаки MELD → samples.pkl для Keras-бейзлайнов")
    ap.add_argument("--text", choices=["word2vec", "fasttext"], required=True)
    ap.add_argument("--audio", choices=["mfcc", "melspectrogram"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-dim", type=int, default=100)
    ap.add_argument("--text-epochs", type=int, default=15)
    args = ap.parse_args()

    train, dev, test = build_classical_meld(
        text_backend=args.text,
        audio_backend=("mfcc" if args.audio == "mfcc" else "mel"),
        text_dim=args.text_dim,
        text_train_epochs=args.text_epochs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preprocess_utils.save_pickle({"train": train, "dev": dev, "test": test}, args.out)
    log.info("Saved %s (train=%d dev=%d test=%d)", args.out, len(train), len(dev), len(test))


if __name__ == "__main__":
    main()
