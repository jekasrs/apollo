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