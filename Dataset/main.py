import pandas as pd
from gensim.models import Word2Vec

from Dataset import MELD_TRAIN_TEXT_PATH, MELD_DEV_TEXT_PATH, MELD_TEST_TEXT_PATH
from Dataset.utils import preprocess_text, word_averaging_list
from sklearn.preprocessing import LabelEncoder

vector_size = 300
window = 5
min_count = 1
epochs = 300


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    df_train = pd.read_csv(MELD_TRAIN_TEXT_PATH)
    df_dev = pd.read_csv(MELD_DEV_TEXT_PATH)
    df_test = pd.read_csv(MELD_TEST_TEXT_PATH)

    print(df_train.shape) # (9989, 11)
    print(df_dev.shape) # (1109, 11)
    print(df_test.shape) # (2610, 11)

    df_train['preprocessed_text'] = df_train['Utterance'].apply(preprocess_text)
    df_dev['preprocessed_text'] = df_train['Utterance'].apply(preprocess_text)
    df_test['preprocessed_text'] = df_train['Utterance'].apply(preprocess_text)

    label_encoder = LabelEncoder()
    df_train['encoded_emotions'] = label_encoder.fit_transform(df_train.Emotion)
    df_dev['encoded_emotions'] = label_encoder.fit_transform(df_dev.Emotion)
    df_test['encoded_emotions'] = label_encoder.fit_transform(df_test.Emotion)

    # Преобразование токенов (строка → список)
    train_tokens = df_train["preprocessed_text"].tolist()
    dev_tokens = df_dev["preprocessed_text"].tolist()
    test_tokens = df_test["preprocessed_text"].tolist()

    #  Обучение Word2Vec
    all_tokens = train_tokens + dev_tokens + test_tokens
    w2v_model = Word2Vec(sentences=all_tokens, vector_size=vector_size, window=window, min_count=min_count)
    w2v_model.build_vocab(all_tokens, update=True)
    w2v_model.train(all_tokens, total_examples=w2v_model.corpus_count, epochs=epochs)

    #  Усреднение векторов
    X_train = word_averaging_list(w2v_model.wv, train_tokens)
    X_dev = word_averaging_list(w2v_model.wv, dev_tokens)
    X_test = word_averaging_list(w2v_model.wv, test_tokens)

    df_train['vector'] = X_train.tolist()
    df_dev['vector'] = X_dev.tolist()
    df_test['vector'] = X_test.tolist()

    # Сохраняем результаты векторизации
    df_train.to_csv(f"meld/vector_train.csv", index=False)
    df_dev.to_csv(f"meld/vector_dev.csv", index=False)
    df_test.to_csv(f"meld/vector_test.csv", index=False)