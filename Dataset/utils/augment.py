"""Аугментация train: шум по каналам модальностей (канал паузы не трогаем)."""

from __future__ import annotations

import random

import torch

from Dataset.utils.constants import AUDIO_FEATURE_DIM


def maybe_augment_input_tensor(
    input_tensor: torch.Tensor,
    modalities: str,
    apply_prob: float,
    std_audio: float,
    std_text: float,
) -> None:
    """
    input_tensor: (B, T, F) где последний канал — пауза.
    Изменяет тензор in-place.
    """
    if apply_prob <= 0 or random.random() > apply_prob:
        return
    x = input_tensor[..., :-1]
    if modalities == "at":
        noise = torch.zeros_like(x)
        noise[..., :AUDIO_FEATURE_DIM] = std_audio * torch.randn_like(x[..., :AUDIO_FEATURE_DIM])
        noise[..., AUDIO_FEATURE_DIM:] = std_text * torch.randn_like(x[..., AUDIO_FEATURE_DIM:])
    elif modalities == "a":
        noise = std_audio * torch.randn_like(x)
    elif modalities == "t":
        noise = std_text * torch.randn_like(x)
    else:
        return
    input_tensor[..., :-1] = x + noise
