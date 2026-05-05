import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
DATASET_PATH = _PKG / "meld" / "text" / "meld.csv"
AUDIO_DIR = _PKG / "meld" / "audio"
# Переопределение: env APOLLO_SAMPLES_PKL или --samples-pkl в train/eval
SAMPLES_PKL = Path(
    os.environ.get("APOLLO_SAMPLES_PKL", str(_PKG / "preprocess" / "samples" / "samples.pkl"))
)