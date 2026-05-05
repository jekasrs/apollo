"""Аугментация train: гауссов шум по каналам модальностей (канал паузы не меняется)."""

from __future__ import annotations

import random
import torch

from dataset.preprocess.utils.constants import AUDIO_FEATURE_DIM


def augment_content_audio_only(
    input_tensor: torch.Tensor, std_audio: float, use_pause: bool = True
) -> None:
    """Добавляет гауссов шум ко всем каналам аудио-признаков."""
    if use_pause:
        x = input_tensor[..., :-1]
        input_tensor[..., :-1] = x + std_audio * torch.randn_like(x)
    else:
        input_tensor[...] = input_tensor + std_audio * torch.randn_like(
            input_tensor
        )


def augment_content_text_only(
    input_tensor: torch.Tensor, std_text: float, use_pause: bool = True
) -> None:
    """Добавляет гауссов шум ко всем каналам текстовых признаков."""
    if use_pause:
        x = input_tensor[..., :-1]
        input_tensor[..., :-1] = x + std_text * torch.randn_like(x)
    else:
        input_tensor[...] = input_tensor + std_text * torch.randn_like(
            input_tensor
        )


def augment_content_audio_text(
    input_tensor: torch.Tensor,
    std_audio: float,
    std_text: float,
    use_pause: bool = True,
) -> None:
    """
    Использовать только в режиме ``at``:
    - аудио (шум ``std_audio``)
    - текст (шум ``std_text``)
    """
    if use_pause:
        x = input_tensor[..., :-1]
        noise = torch.zeros_like(x)
        noise[..., :AUDIO_FEATURE_DIM] = std_audio * torch.randn_like(
            x[..., :AUDIO_FEATURE_DIM]
        )
        noise[..., AUDIO_FEATURE_DIM:] = std_text * torch.randn_like(
            x[..., AUDIO_FEATURE_DIM:]
        )
        input_tensor[..., :-1] = x + noise
    else:
        x = input_tensor
        noise = torch.zeros_like(x)
        noise[..., :AUDIO_FEATURE_DIM] = std_audio * torch.randn_like(
            x[..., :AUDIO_FEATURE_DIM]
        )
        noise[..., AUDIO_FEATURE_DIM:] = std_text * torch.randn_like(
            x[..., AUDIO_FEATURE_DIM:]
        )
        input_tensor[...] = x + noise


def maybe_augment_input_tensor(
    input_tensor: torch.Tensor,
    modalities: str,
    apply_prob: float,
    std_audio: float,
    std_text: float,
    use_pause: bool = True,
) -> None:
    """
    С вероятностью ``apply_prob`` вызывает одну из функций аугментации по ``modalities``.
    При ``apply_prob <= 0`` или неудачном броске выходит без изменений.
    """
    if apply_prob <= 0 or random.random() > apply_prob:
        return
    if modalities == "a":
        augment_content_audio_only(input_tensor, std_audio, use_pause=use_pause)
    elif modalities == "t":
        augment_content_text_only(input_tensor, std_text, use_pause=use_pause)
    elif modalities == "at":
        augment_content_audio_text(
            input_tensor, std_audio, std_text, use_pause=use_pause
        )
