"""Единый выбор устройства: NVIDIA GPU → Apple MPS → CPU."""

from __future__ import annotations

import torch


def preferred_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
