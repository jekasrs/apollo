"""Нормализация паузы log(1+pause) по статистике train (mu, std)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Tuple


def compute_pause_norm_stats(samples: Iterable[Any]) -> Tuple[float, float]:
    vals = [math.log1p(max(0.0, float(getattr(s, "pause", 0.0)))) for s in samples]
    if not vals:
        return 0.0, 1.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var + 1e-8)
    if std < 1e-8:
        std = 1.0
    return mu, std
