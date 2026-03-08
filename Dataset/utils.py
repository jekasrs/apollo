import string

import numpy as np
from nltk import word_tokenize, WordNetLemmatizer
import gensim

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'while', 'both', 'each', 'other',
    'again', 'only', 'between', 'during', 'before', 'after', 'above', 'below', 'from', 'into',
    'over', 'under', 'through', 'in', 'out', 'on', 'off', 'up', 'down', 'at', 'by', 'with',
    'without', 'to', 'of', 'for', 'once', 'then', 'than', 'too', 'such', 'so', 'just',
    'further', 'more', 'most', 'few', 'some', 'any', 'all', 'very', 'own', 'same', 'about',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'
}

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Приведение текста к нижнему регистру
    text = text.lower()
    # Удаление лишних символов
    text = ''.join([char for char in text if char not in string.punctuation and char not in '’—‘”“' and char.isalpha() or char == ' '])
    # Удаление чисел
    text = ''.join([char for char in text if not char.isdigit()])
    # Инициализация объекта для обработки текста
    tokens = word_tokenize(text)
    # Токенизация и Лемматизация текста и удаление стоп-слов
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if lemmatizer.lemmatize(token) not in STOP_WORDS]
    # Объединение обработанных слов
    lemmatized_tokens = ' '.join(lemmatized_tokens)
    # Возвращение предобработанного текста
    return lemmatized_tokens


def word_averaging(model, words):
    mean = []

    for word in words:
        if word in model.key_to_index.keys():
            mean.append(model[word])

    # Если ни одно слово из текста не найдено
    if not mean:
        return np.zeros(model.vector_size)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32) # Усреднение и нормализация
    return mean


def word_averaging_list(model, text_list):
    return np.vstack([word_averaging(model, comment_text) for comment_text in text_list ])
