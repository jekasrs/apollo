"""
Препроцесс MELD: CSV → Sample (текст, аудио-вектор, utterance_id, пауза), сплит, pickle.
"""

import logging
from typing import Any

from tqdm import tqdm

from dataset import DATASET_PATH, AUDIO_DIR, SAMPLES_PKL
from dataset.models.Sample import Sample
from dataset.preprocess.utils import utils as preprocess_utils
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils.hf_mpnet_encoder import make_text_embedder_for_preprocess
from dataset.preprocess.utils.Wav2VecEmbedder import Wav2VecEmbedder

log = logging.getLogger(__name__)
batch_size = dataset_constants.WAV2VEC_BATCH_SIZE


def _build_encoders() -> tuple[Any, Wav2VecEmbedder]:
    text_model = make_text_embedder_for_preprocess()
    fin = dataset_constants.get_finetuned_wav2vec_dir()
    if fin:
        log.info("Аудио: дообученный Wav2Vec2 из %s", fin)
    audio_model = Wav2VecEmbedder(
        dataset_constants.WAV2VEC_MODEL_NAME,
        finetuned_path=fin,
    )
    return text_model, audio_model


def get_meld():
    text_model, audio_model = _build_encoders()
    df = preprocess_utils.load_dataset(DATASET_PATH, AUDIO_DIR)
    samples = []
    pending = []

    prev_did = None
    utt_idx = 0

    def flush():
        nonlocal pending
        if not pending:
            return
        texts = [p["text"] for p in pending]
        embs = text_model.encode_batch(texts)
        feats = audio_model.encode_batch([p["audio"] for p in pending])
        for p, emb, audio_feat in zip(pending, embs, feats):
            samples.append(
                Sample(
                    utterance_id=p["utterance_id"],
                    text=p["text"],
                    audio_path=p["audio_path"],
                    label=p["label"],
                    dialogue_id=p["dialogue_id"],
                    start=p["start"],
                    end=p["end"],
                    embeddings=emb,
                    speaker_name=str(p["speaker"]),
                    audio_features=audio_feat,
                )
            )
        pending = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MELD preprocess"):
        did = row["dialogue_id"]
        if prev_did != did:
            utt_idx = 0
        prev_did = did

        text = preprocess_utils.clean_text(row["utterance"], remove_stopwords=False)
        audio, sr = preprocess_utils.load_audio_segment(row["path_to_audio"])
        audio = preprocess_utils.normalize_audio(audio)

        pending.append(
            {
                "utterance_id": utt_idx,
                "text": text,
                "audio_path": row["path_to_audio"],
                "label": row["emotion"],
                "dialogue_id": row["dialogue_id"],
                "speaker": row["speaker"],
                "start": row["start"],
                "end": row["end"],
                "audio": audio,
            }
        )
        utt_idx += 1

        if len(pending) >= batch_size:
            flush()

    flush()

    preprocess_utils.assign_pause_until_next_in_dialogues(samples)

    return preprocess_utils.split_samples_by_dialogue(
        samples,
        test_size=dataset_constants.TEST_SIZE,
        dev_size=dataset_constants.DEV_SIZE,
        random_state=dataset_constants.RANDOM_STATE,
    )


def main():
    train, dev, test = get_meld()
    pause_mu, pause_std = preprocess_utils.compute_pause_norm_stats(train)
    pm, ps = float(pause_mu), float(pause_std)
    for s in train + dev + test:
        s.pause_norm_mu = pm
        s.pause_norm_std = ps
    data = {"train": train, "dev": dev, "test": test}
    preprocess_utils.save_pickle(data, SAMPLES_PKL)


if __name__ == "__main__":
    main()
