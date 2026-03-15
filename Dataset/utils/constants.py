from pathlib import Path
# пути
DATASET_PATH = Path("meld/meld.csv")
AUDIO_DIR = Path("/Users/evsmirnovalek/PycharmProjects/apollo/Dataset/meld/audio")
SAMPLES_PATH = Path("models/samples.pkl")

# аудио настройки
SAMPLE_RATE = 16000
N_MFCC = 13

# текст настройки
REMOVE_STOPWORDS = False

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