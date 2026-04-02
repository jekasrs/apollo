# Констатнты для дата-сетов

# аудио: препроцессинг через Wav2Vec2 (см. Dataset/preprocess.py, Dataset/utils/wav2vec_features.py)
SAMPLE_RATE = 16000
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"
# Размерность скрытого состояния модели (для wav2vec2-base = 768; при смене модели обновить).
AUDIO_FEATURE_DIM = 768
WAV2VEC_BATCH_SIZE = 8

# SentenceTransformer для preprocess (768 измерений); после смены — заново: python Dataset/preprocess.py
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# эмоции
EMOTION_MAP = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6
}

TEST_SIZE: float = 0.2
DEV_SIZE: float = 0.25
RANDOM_STATE: int = 42

TEXT_EMBED_DIM = 768

DIMS = {
    "a": AUDIO_FEATURE_DIM,
    "t": TEXT_EMBED_DIM,
    "at": TEXT_EMBED_DIM + AUDIO_FEATURE_DIM,
}

# Канал паузы в конце входного вектора (после модальностей).
PAUSE_FEATURE_DIM = 1
