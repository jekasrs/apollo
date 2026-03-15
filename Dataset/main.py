from sklearn.model_selection import train_test_split

from Dataset.models.Sample import Sample
from Dataset.utils import constants as dataset_constants
from Dataset.utils.io_utils import save_pickle
from Dataset.utils import utils as dataset_utils
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("paraphrase-distilroberta-base-v1")


def get_meld():
    df = dataset_utils.load_dataset(dataset_constants.DATASET_PATH, dataset_constants.AUDIO_DIR)
    samples = []

    prev_end = None
    prev_dialogue_id = None

    for _, row in df.iterrows():
        if prev_dialogue_id != row["dialogue_id"]:
            prev_dialogue_id = row["dialogue_id"]
            prev_end = None

        text = dataset_utils.clean_text(row["utterance"], remove_stopwords=False)
        embedding = dataset_utils.extract_embeddings(sentence=text, model=model)
        audio, sr = dataset_utils.load_audio_segment(
            row["path_to_audio"],
            row["start"],
            row["end"]
        )

        audio = dataset_utils.normalize_audio(audio)
        mfcc = dataset_utils.extract_mfcc(audio, sr)

        sample = Sample(
            text=text,
            audio_path=row["path_to_audio"],
            label=row["emotion"],
            dialogue_id=row["dialogue_id"],
            speaker_id=row["Speaker"],
            start=row["start"],
            end=row["end"],
            prev_end=prev_end,
            embeddings=embedding,
            mfcc=mfcc
        )

        samples.append(sample)
        prev_end = row["end"]

    temp, test = train_test_split(
        samples,
        test_size=dataset_constants.TEST_SIZE,
        random_state=dataset_constants.RANDOM_STATE
    )

    dev_size_relative = dataset_constants.DEV_SIZE / (1 - dataset_constants.TEST_SIZE)
    train, dev = train_test_split(
        temp,
        test_size=dev_size_relative,
        random_state=dataset_constants.RANDOM_STATE
    )

    return train, dev, test


if __name__ == '__main__':
    train, dev, test = get_meld()
    data = {"train": train, "dev": dev, "test": test}
    save_pickle(data, f"{dataset_constants.SAMPLES_PATH}")
