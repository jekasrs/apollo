import math
import random
import torch


def _sample_audio_vec(sample) -> torch.Tensor:
    """Новые pickle: audio_features (Wav2Vec2). Старые: mfcc — только если размерность совпадает с моделью."""
    if hasattr(sample, "audio_features") and sample.audio_features is not None:
        return torch.as_tensor(sample.audio_features, dtype=torch.float32)
    if hasattr(sample, "mfcc") and sample.mfcc is not None:
        return torch.as_tensor(sample.mfcc, dtype=torch.float32)
    raise AttributeError(
        "Sample должен содержать audio_features (после preprocess с Wav2Vec2) или устаревшее поле mfcc"
    )


class Dataset:
    def __init__(self, samples, batch_size, modalities, dataset_embedding_dims) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.modalities = modalities
        self.embedding_dim = dataset_embedding_dims

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        # One timestep per sample: текст (BERT/ST) + аудио (Wav2Vec2 pooled).
        text_len_tensor = torch.ones(batch_size, dtype=torch.long)
        input_tensor = torch.zeros((batch_size, 1, self.embedding_dim))
        speaker_tensor = torch.zeros((batch_size, 1), dtype=torch.long)
        labels = []
        utterance_texts = []
        for i, s in enumerate(samples):
            utterance_texts.append(s.text)
            t = torch.as_tensor(s.embeddings, dtype=torch.float32)
            a = _sample_audio_vec(s)
            if self.modalities == "at":
                feat = torch.cat((a, t))
            elif self.modalities == "a":
                feat = a
            elif self.modalities == "t":
                feat = t
            else:
                raise ValueError(f"Unknown modalities: {self.modalities}")

            input_tensor[i, 0, :] = feat
            speaker_tensor[i, 0] = int(s.speaker_id)

            labels.append(s.label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "input_tensor": input_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterance_texts,
        }
        return data

    def shuffle(self):
        random.shuffle(self.samples)
