import re
import pandas as pd
import librosa
import numpy as np

from nltk.stem import WordNetLemmatizer
from pathlib import Path

from Dataset.utils.constants import EMOTION_MAP, SPEAKER_MAP

lemmatizer = WordNetLemmatizer()
stop_words = {}


def load_audio_segment(path, sr=16000):
    y, sr = librosa.load(path, sr=sr)
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

    # Замена текстовых эмоций на числовые
    df["emotion"] = df["emotion"].map(EMOTION_MAP)
    df["speaker"] = df["speaker"].map(SPEAKER_MAP)
    df = df.dropna(subset=["emotion", "speaker"])

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


def time_to_seconds(time) -> float:
    """Convert MELD-style timestamps ('HH:MM:SS,mmm' or 'HH:MM:SS') to seconds."""
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
