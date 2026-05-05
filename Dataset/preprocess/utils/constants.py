"""Константы для обработки дата сета MELD."""

import os
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    """dataset/preprocess/utils/constants.py → корень репозитория."""
    return Path(__file__).resolve().parents[3]

# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
# База для дообучения под MELD (тот же семейства MPNet, 768d)
HUGGINGFACE_MPNET_BASE = "microsoft/mpnet-base"
TEXT_EMBED_DIM = 768  # 768 размерность текстового эмбеддинга

# https://huggingface.co/facebook/wav2vec2-base
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"
AUDIO_FEATURE_DIM = 768  # размерность пулинга Wav2Vec2
WAV2VEC_BATCH_SIZE = 8  # размер батча сегментов при кодировании аудио

# После `dataset/finetune/finetune_*.py` задайте путь: env `APOLLO_FINETUNED_TEXT` / `APOLLO_FINETUNED_WAV2VEC`;
# если env пуст, подхватывается `results/encoders/finetune_mpnet_meld` при наличии каталога.
# Либо укажите вручную (имеет приоритет над env, если env пуст):
FINETUNED_TEXT_DIR_OVERRIDE: Optional[str] = None
FINETUNED_WAV2VEC_DIR_OVERRIDE: Optional[str] = None


def _env_path(key: str) -> Optional[str]:
    v = os.environ.get(key, "").strip()
    return v if v else None


def _default_finetuned_mpnet_dir() -> Optional[str]:
    p = _repo_root() / "results" / "encoders" / "finetune_mpnet_meld"
    return str(p) if p.is_dir() else None


def _default_finetuned_wav2vec_dir() -> Optional[str]:
    p = _repo_root() / "results" / "encoders" / "finetune_wav2vec_meld"
    return str(p) if p.is_dir() else None


def get_finetuned_text_dir() -> Optional[str]:
    return (
        _env_path("APOLLO_FINETUNED_TEXT") or FINETUNED_TEXT_DIR_OVERRIDE or _default_finetuned_mpnet_dir()
    )


def get_finetuned_wav2vec_dir() -> Optional[str]:
    return (
        _env_path("APOLLO_FINETUNED_WAV2VEC")
        or FINETUNED_WAV2VEC_DIR_OVERRIDE
        or _default_finetuned_wav2vec_dir()
    )


SAMPLE_RATE = 16000  # 16000 Гц для librosa.load

# Канал паузы в конце входного вектора
PAUSE_FEATURE_DIM = 1  # скаляр паузы (после конката модальностей)

RANDOM_STATE: int = 42  # seed sklearn train_test_split
TEST_SIZE: float = 0.2  # 0.2 доля test по dialogue_id
DEV_SIZE: float = 0.25  # 0.25 доля dev в части без test

EMOTION_MAP = { # 7 классов
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}