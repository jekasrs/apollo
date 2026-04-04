from pathlib import Path

_PKG = Path(__file__).resolve().parent
DATASET_PATH = _PKG / "meld" / "text" / "meld.csv"
AUDIO_DIR = _PKG / "meld" / "audio"
SAMPLES_PKL = _PKG / "preprocess" / "samples" / "samples.pkl"