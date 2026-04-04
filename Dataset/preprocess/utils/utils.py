from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Any, Tuple

import librosa
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from dataset.preprocess.utils.constants import EMOTION_MAP, SAMPLE_RATE

lemmatizer = WordNetLemmatizer()
stop_words = {}


def load_audio_segment(path, sr=SAMPLE_RATE):
    """Загружает wav-файл реплики, приводит к моно и частоте дискретизации ``sr`` (по умолчанию 16 kHz)."""
    y, sr = librosa.load(path, sr=sr)
    return y, sr


def load_dataset(csv_path, audio_dir):
    """
    Читает CSV MELD, строит путь к каждому ``.wav``, переименовывает колонки под код,
    переводит ``start``/``end`` в секунды, мапит эмоции через ``EMOTION_MAP``, спикера — в строку.
    Строки без известной эмоции отбрасываются. Возвращает готовый ``DataFrame`` для итерации в preprocess.
    """
    df = pd.read_csv(csv_path)
    audio_dir = Path(audio_dir)

    df["path_to_audio"] = df.apply(
        lambda row: audio_dir / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}_seas{row['Season']}.wav",
        axis=1
    )

    df = df.rename(columns={
        "Utterance": "utterance",
        "Speaker": "speaker",
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Season": "season",
        "Emotion": "emotion",
        "StartTime": "start",
        "EndTime": "end",
    })

    df["start"] = df["start"].apply(time_to_seconds)
    df["end"] = df["end"].apply(time_to_seconds)

    df["emotion"] = df["emotion"].map(EMOTION_MAP)
    df["speaker"] = df["speaker"].astype(str)
    df = df.dropna(subset=["emotion"])

    return df


def clean_text(text: str, remove_stopwords=True):
    """
    Нормализует текст реплики: нижний регистр, удаление пунктуации и цифр, лемматизация (WordNet).
    Опционально отфильтровывает стоп-слова (словарь stop_words).

    Возвращает одну строку токенов через пробел.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)


def normalize_audio(y):
    """Приводит амплитуду сигнала к единичной норме (peak normalization), чтобы громкость не доминировала."""
    return librosa.util.normalize(y)


def extract_embeddings(sentence, model):
    """
    Кодирует предложение в вектор признаков через переданную модель (обычно SentenceTransformer).
    Возвращает numpy-вектор фиксированной размерности.
    """
    return model.encode(sentence)


def time_to_seconds(time) -> float:
    """Преобразует отметки времени MELD (``'HH:MM:SS,mmm'`` или ``'HH:MM:SS'``) в секунды (float)."""
    if pd.isna(time):
        return float("nan")
    time_str = str(time).strip().replace(",", ".")
    parts = re.split(r"[:.]", time_str)
    if len(parts) == 4:
        h, m, s_int, frac = parts
        return float(h) * 3600 + float(m) * 60 + float(s_int) + float(f"0.{frac}")
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    raise ValueError(f"Unrecognized time format: {time!r}")


def assign_pause_until_next_in_dialogues(samples) -> None:
    """
    По каждому dialogue_id: сортировка по времени; первая реплика pause=0;
    внутренние — max(0, start_след − end_тек); последняя — 0.
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s.dialogue_id].append(s)
    for utts in groups.values():
        utts.sort(key=lambda x: (x.start, x.end))
        n = len(utts)
        if n == 0:
            continue
        utts[0].pause = 0.0
        if n == 1:
            continue
        utts[-1].pause = 0.0
        for i in range(1, n - 1):
            gap = float(utts[i + 1].start) - float(utts[i].end)
            utts[i].pause = max(0.0, gap)


def compute_pause_norm_stats(samples: Iterable[Any]) -> Tuple[float, float]:
    """Нормализация паузы: μ и σ по log1p(pause) на train."""
    vals = [math.log1p(max(0.0, float(getattr(s, "pause", 0.0)))) for s in samples]
    if not vals:
        return 0.0, 1.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var + 1e-8)
    if std < 1e-8:
        std = 1.0
    return mu, std


def split_samples_by_dialogue(samples, test_size, dev_size, random_state):
    """Сплит по dialogue_id: train / dev / test без пересечения диалогов."""
    groups = defaultdict(list)
    for s in samples:
        groups[s.dialogue_id].append(s)
    dialogue_ids = list(groups.keys())
    train_ids, test_ids = train_test_split(
        dialogue_ids,
        test_size=test_size,
        random_state=random_state,
    )
    dev_rel = dev_size / (1 - test_size)
    train_ids, dev_ids = train_test_split(
        train_ids,
        test_size=dev_rel,
        random_state=random_state,
    )
    train = [utt for did in train_ids for utt in groups[did]]
    dev = [utt for did in dev_ids for utt in groups[did]]
    test = [utt for did in test_ids for utt in groups[did]]
    return train, dev, test

import pickle
from pathlib import Path


def save_pickle(obj, path):
    """Сохраняет объект в pickle файл."""
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Загружает объект из pickle файла."""
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj