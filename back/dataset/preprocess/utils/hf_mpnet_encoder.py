"""MPNet, дообученный на MELD: извлечение 768-d эмбеддингов из backbone (как в Sentence-BERT)."""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import torch
from transformers import AutoTokenizer, MPNetForSequenceClassification

from dataset.preprocess.utils.constants import TEXT_ENCODER_MAX_LENGTH, get_finetuned_text_dir
from dataset.preprocess.utils.torch_device import preferred_torch_device

log = logging.getLogger(__name__)


class FinetunedMPNetTextEncoder:
    """
    Грузит ``MPNetForSequenceClassification`` и для ``encode_batch`` применяет
    mean pooling по subword-маске к последнему hidden слою ``model.mpnet``.
    """

    def __init__(self, model_dir: str, device: torch.device | None = None, max_length: int | None = None) -> None:
        self.device = device or preferred_torch_device()
        self.max_length = int(max_length) if max_length is not None else TEXT_ENCODER_MAX_LENGTH
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        full = MPNetForSequenceClassification.from_pretrained(model_dir)
        self.backbone: Any = full.mpnet.to(self.device)
        self.backbone.eval()
        self._hidden = full.config.hidden_size

    @torch.inference_mode()
    def forward_pool(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self._hidden, device=self.device)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.backbone(**enc)
        h = out.last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(dtype=h.dtype)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        return pooled

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._hidden), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t = self.forward_pool(batch)
            chunks.append(t.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


def make_text_embedder_for_preprocess() -> "FinetunedMPNetTextEncoder | SentenceTransformerBatchAdapter":
    from sentence_transformers import SentenceTransformer

    from dataset.preprocess.utils import constants as c

    path = c.get_finetuned_text_dir()
    if path:
        log.info("Текстовые эмбеддинги: дообученный MPNet (%s)", path)
        return FinetunedMPNetTextEncoder(path)
    log.info("Текстовые эмбеддинги: SentenceTransformer (%s)", c.SENTENCE_TRANSFORMER_MODEL)
    return SentenceTransformerBatchAdapter(SentenceTransformer(c.SENTENCE_TRANSFORMER_MODEL))


def load_text_encoder_for_preprocess() -> FinetunedMPNetTextEncoder:
    p = get_finetuned_text_dir()
    if not p:
        raise ValueError("Задайте APOLLO_FINETUNED_TEXT или constants.FINETUNED_TEXT_DIR_OVERRIDE")
    return FinetunedMPNetTextEncoder(p)


class SentenceTransformerBatchAdapter:
    """Обертка: единый интерфейс ``encode_batch`` для SentenceTransformer."""

    def __init__(self, st_model: Any) -> None:
        self._m = st_model

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        v = self._m.encode(
            texts,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        if isinstance(v, list):
            return np.stack(v, axis=0).astype(np.float32)
        return np.asarray(v, dtype=np.float32)
