from pathlib import Path

# Paths relative to this package (Dataset/) so preprocess/train work from project root
_PKG = Path(__file__).resolve().parent
DATASET_PATH = _PKG / "meld" / "meld.csv"
AUDIO_DIR = _PKG / "meld" / "audio"
# train.py loads: Dataset / SAMPLES_PATH
SAMPLES_PATH = Path("meld/samples.pkl")