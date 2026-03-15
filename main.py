import logging

from Dataset.utils.constants import SAMPLES_PATH
from Dataset.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Загрузка данных
    data = load_pickle(f"Dataset/{SAMPLES_PATH}")
    logging.log(logging.INFO, f"Loaded data set MELD")

    test = data.get('train')
    train = data.get('train')
    dev = data.get('dev')
    logging.log(logging.INFO, f"train array len={len(train)}, dev array len={len(dev)}, test array len={len(test)}")
