"""Пулинг эмбеддингов реплики через предобученный Wav2Vec2 (без fine-tune на этапе препроцесса)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2VecEmbedder:
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """
        waveforms: список 1D float32/float64, sample rate = 16 kHz (как в SAMPLE_RATE).
        Возвращает (B, hidden_size) — среднее по времени с учётом padding mask.
        """
        if not waveforms:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        out = self.model(input_values, attention_mask=attention_mask)
        h = out.last_hidden_state
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)
        return pooled.cpu().numpy().astype(np.float32)
