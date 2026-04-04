"""Константы для обработки дата сета MELD."""

# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
TEXT_EMBED_DIM = 768  # 768 размерность текстового эмбеддинга

# https://huggingface.co/facebook/wav2vec2-base
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"
AUDIO_FEATURE_DIM = 768  # размерность пулинга Wav2Vec2
WAV2VEC_BATCH_SIZE = 8  # размер батча сегментов при кодировании аудио
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