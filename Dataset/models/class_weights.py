"""Class weights for imbalanced emotion labels (balanced inverse frequency)."""
import numpy as np
import torch


def compute_class_weights_from_samples(samples, num_classes, device=None) -> torch.Tensor:
    """
    sklearn-style 'balanced' weights: n_samples / (n_classes * count_k).
    Classes absent in `samples` get count floored to 1 to avoid division by zero.
    """
    if not samples:
        return torch.ones(num_classes, dtype=torch.float32, device=device)
    y = np.array([int(s.label) for s in samples], dtype=np.int64)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = len(y)
    weights = n / (num_classes * counts)
    t = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t
