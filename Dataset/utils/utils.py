import re
import pandas as pd
import librosa
import numpy as np

from nltk.stem import WordNetLemmatizer
from pathlib import Path

from Dataset.utils.constants import EMOTION_MAP

lemmatizer = WordNetLemmatizer()
stop_words = {}


def load_audio_segment(path, start, end, sr=16000):
    y, sr = librosa.load(path, sr=sr)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    # segment = y[start_sample:end_sample]
    return y, sr


def load_dataset(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    audio_dir = Path(audio_dir)

    df["path_to_audio"] = df.apply(
        lambda row: audio_dir / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}_seas{row['Season']}.wav",
        axis=1
    )

    df = df.rename(columns={
        "Utterance": "utterance",
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Season": "season",
        "Emotion": "emotion",
        "StartTime": "start",
        "EndTime": "end",
    })

    df["start"] = df["start"].apply(time_to_seconds)
    df["end"] = df["end"].apply(time_to_seconds)

    # Замена текстовых эмоций на числовые
    df["emotion"] = df["emotion"].map(EMOTION_MAP)

    return df


def clean_text(text: str, remove_stopwords=True):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)


def normalize_audio(y):
    return librosa.util.normalize(y)


def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])
    return features


def extract_embeddings(sentence, model):
    return model.encode(sentence)


def time_to_seconds(time: str):
    """Преобразует 'HH:MM:SS,mmm' в секунды."""
    time_str = time.replace(",", ".")
    h, m, s = map(float, re.split("[:.]", time_str)[:3])
    ms = float("0." + time_str.split(".")[-1])
    return h * 3600 + m * 60 + s + ms
