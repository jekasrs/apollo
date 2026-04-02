"""Class weights для дисбаланса: effective number of samples (Cui et al., class-balanced loss)."""
import numpy as np
import torch


def compute_class_weights_from_samples(
    samples,
    num_classes,
    device=None,
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Веса по effective number: сильнее поднимают редкие классы, чем простой balanced.
    beta ближе к 1 — мягче; 0.999 хорошо для длинных хвостов в MELD.
    """
    if not samples:
        return torch.ones(num_classes, dtype=torch.float32, device=device)
    y = np.array([int(s.label) for s in samples], dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_classes
    t = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t
