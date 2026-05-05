"""
Дублирование train-диалогов MELD, содержащих заданные метки, с новыми dialogue_id
(чтобы в Dataset не слипались реплики разных копий).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from dataset.models.Sample import Sample

__all__ = ["duplicate_train_dialogues_by_labels"]


def _clone_utterance(s: Sample, new_dialogue_id: str, copy_idx: int) -> Sample:
    uid = s.utterance_id
    if copy_idx:
        uid = f"{uid}__od{copy_idx}"
    return Sample(
        uid,
        s.text,
        s.audio_path,
        s.label,
        new_dialogue_id,
        s.start,
        s.end,
        s.embeddings,
        s.audio_features,
        s.speaker_name,
        s.pause,
        speaker_id=s.speaker_id,
        pause_norm_mu=s.pause_norm_mu,
        pause_norm_std=s.pause_norm_std,
    )


def duplicate_train_dialogues_by_labels(
    samples: Sequence[Sample],
    label_ids: set[int],
    n_extra: int,
) -> tuple[list[Sample], int, int]:
    """
    Для каждого диалога, где есть хотя бы одна реплика с label in label_ids,
    добавить ``n_extra`` полных копий всех реплик (новый dialogue_id у копий).

    Returns (новый_список, число_добавленных_реплик, число_затронутых_диалогов)
    """
    n_extra = int(n_extra)
    if n_extra <= 0 or not label_ids:
        return list(samples), 0, 0

    by_d: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_d[s.dialogue_id].append(s)

    for utts in by_d.values():
        utts.sort(key=lambda x: (x.start, x.end))

    out: list[Sample] = list(samples)
    n_added = 0
    n_dialogs = 0
    dup_key = 0
    for did, group in by_d.items():
        if not any(s.label in label_ids for s in group):
            continue
        n_dialogs += 1
        for k in range(n_extra):
            dup_key += 1
            new_did = f"{did}__rarex{k}_{dup_key}"
            for s in group:
                out.append(_clone_utterance(s, new_did, dup_key))
                n_added += 1
    return out, n_added, n_dialogs
