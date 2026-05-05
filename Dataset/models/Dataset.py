import math
import random
from collections import defaultdict

import torch

from dataset.preprocess.utils.constants import PAUSE_FEATURE_DIM
from models.apollo.utils.functions import BATCH_KEY_UTTERANCE_TEXTS


def _sample_audio_vec(sample) -> torch.Tensor:
    if hasattr(sample, "audio_features") and sample.audio_features is not None:
        return torch.as_tensor(sample.audio_features, dtype=torch.float32)
    if hasattr(sample, "mfcc") and sample.mfcc is not None:
        return torch.as_tensor(sample.mfcc, dtype=torch.float32)
    raise AttributeError(
        "Sample должен содержать audio_features или устаревшее поле mfcc"
    )


def _speaker_key(sample) -> str:
    sn = getattr(sample, "speaker_name", None)
    if sn is not None and str(sn).strip():
        return str(sn)
    sid = getattr(sample, "speaker_id", None)
    if sid is not None:
        return f"__legacy_id_{int(sid)}__"
    return "__unknown__"


def _local_speaker_ids(utterances):
    """Порядок первого появления имени → 0, 1, … внутри диалога."""
    key_to_local = {}
    out = []
    for s in utterances:
        k = _speaker_key(s)
        if k not in key_to_local:
            key_to_local[k] = len(key_to_local)
        out.append(key_to_local[k])
    return out


def _group_samples_into_dialogues(samples):
    d = defaultdict(list)
    for s in samples:
        d[s.dialogue_id].append(s)
    groups = []
    for utts in d.values():
        utts.sort(key=lambda x: (x.start, x.end))
        groups.append(utts)
    return groups


class Dataset:
    """
    Батчи по диалогам. При ``use_pause=True`` последний канал входа — нормализованная пауза
    (log1p + z-score по train); при ``False`` подаются только признаки модальностей.
    """

    def __init__(
        self,
        samples,
        dialogues_per_batch: int,
        modalities: str,
        modality_feature_dim: int,
        pause_mu: float,
        pause_std: float,
        augment: bool = False,
        use_pause: bool = True,
    ) -> None:
        self.modalities = modalities
        self.modality_feature_dim = modality_feature_dim
        self.use_pause = use_pause
        self.embedding_dim = modality_feature_dim + (
            PAUSE_FEATURE_DIM if use_pause else 0
        )
        self.pause_mu = float(pause_mu)
        self.pause_std = float(pause_std) if float(pause_std) > 1e-8 else 1.0
        self.dialogues_per_batch = max(1, int(dialogues_per_batch))
        self.augment = augment
        self._dialogue_groups = _group_samples_into_dialogues(samples)
        n = len(self._dialogue_groups)
        self.num_batches = math.ceil(n / self.dialogues_per_batch) if n > 0 else 0

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        start = index * self.dialogues_per_batch
        end = min(start + self.dialogues_per_batch, len(self._dialogue_groups))
        return self._dialogue_groups[start:end]

    def _norm_pause(self, sample) -> float:
        lp = math.log1p(max(0.0, float(getattr(sample, "pause", 0.0))))
        return (lp - self.pause_mu) / self.pause_std

    def padding(self, dialogue_batch):
        d_size = len(dialogue_batch)
        max_len = max(len(g) for g in dialogue_batch)
        input_tensor = torch.zeros((d_size, max_len, self.embedding_dim))
        speaker_tensor = torch.zeros((d_size, max_len), dtype=torch.long)
        labels = []
        utterance_texts = []

        for di, g in enumerate(dialogue_batch):
            locals_ = _local_speaker_ids(g)
            for j, s in enumerate(g):
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
                if self.use_pause:
                    p = torch.tensor([self._norm_pause(s)], dtype=torch.float32)
                    full = torch.cat([feat, p])
                else:
                    full = feat
                input_tensor[di, j, :] = full
                speaker_tensor[di, j] = locals_[j]
                labels.append(s.label)
                utterance_texts.append(s.text)

        text_len_tensor = torch.tensor([len(g) for g in dialogue_batch], dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        data = {
            "text_len_tensor": text_len_tensor,
            "input_tensor": input_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            BATCH_KEY_UTTERANCE_TEXTS: utterance_texts,
        }
        return data

    def shuffle(self):
        random.shuffle(self._dialogue_groups)
