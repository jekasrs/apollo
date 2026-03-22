from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from Dataset import DATASET_PATH, AUDIO_DIR
from Dataset.models.Sample import Sample
from Dataset.utils import constants as dataset_constants
from Dataset.utils.io_utils import save_pickle
from Dataset.utils import utils as dataset_utils
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("paraphrase-distilroberta-base-v1")


def _split_samples_by_dialogue(samples, test_size, dev_size, random_state):
    """Split so train/dev/test do not share the same dialogue_id."""
    groups = defaultdict(list)
    for s in samples:
        groups[s.dialogue_id].append(s)
    dialogue_ids = list(groups.keys())
    train_ids, test_ids = train_test_split(
        dialogue_ids,
        test_size=test_size,
        random_state=random_state,
    )
    dev_rel = dev_size / (1 - test_size)
    train_ids, dev_ids = train_test_split(
        train_ids,
        test_size=dev_rel,
        random_state=random_state,
    )

    def flatten(ids):
        return [utt for did in ids for utt in groups[did]]

    return flatten(train_ids), flatten(dev_ids), flatten(test_ids)


def get_meld():
    df = dataset_utils.load_dataset(DATASET_PATH, AUDIO_DIR)
    samples = []

    prev_end = None
    prev_dialogue_id = None

    for _, row in df.iterrows():
        if prev_dialogue_id != row["dialogue_id"]:
            prev_dialogue_id = row["dialogue_id"]
            prev_end = None

        text = dataset_utils.clean_text(row["utterance"], remove_stopwords=False)
        embedding = dataset_utils.extract_embeddings(sentence=text, model=model)
        audio, sr = dataset_utils.load_audio_segment(row["path_to_audio"])

        audio = dataset_utils.normalize_audio(audio)
        mfcc = dataset_utils.extract_mfcc(audio, sr)

        sample = Sample(
            text=text,
            audio_path=row["path_to_audio"],
            label=row["emotion"],
            dialogue_id=row["dialogue_id"],
            speaker_id=row["speaker"],
            start=row["start"],
            end=row["end"],
            prev_end=prev_end,
            embeddings=embedding,
            mfcc=mfcc
        )

        samples.append(sample)
        prev_end = row["end"]

    return _split_samples_by_dialogue(
        samples,
        test_size=dataset_constants.TEST_SIZE,
        dev_size=dataset_constants.DEV_SIZE,
        random_state=dataset_constants.RANDOM_STATE,
    )


def main():
    train, dev, test = get_meld()
    data = {"train": train, "dev": dev, "test": test}
    out_path = Path(__file__).resolve().parent / "meld" / "samples.pkl"
    save_pickle(data, out_path)


if __name__ == '__main__':
    main()
