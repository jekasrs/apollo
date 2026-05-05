"""
Инференс Apollo по одному диалогу: эмбеддинги реплик (sentence-transformers) и
Sample → Dataset (один батч) → модель. Чекпоинт определяет use_pause и модальности.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from dataset.models.Dataset import Dataset
from dataset.models.Sample import Sample
from dataset.preprocess.utils import constants as dataset_constants
from models.apollo.Apollo import Apollo
from models.apollo.utils.functions import batch_to_device
from models.apollo.utils import constants as apollo_constants

logger = logging.getLogger(__name__)


def _state_dict_from_checkpoint(ckpt: dict) -> dict | None:
    if "best_state" in ckpt and isinstance(ckpt["best_state"], dict):
        return ckpt["best_state"]
    if "state_dict" not in ckpt:
        return None
    inner = ckpt["state_dict"]
    if isinstance(inner, torch.nn.Module):
        return inner.state_dict()
    if isinstance(inner, dict):
        return inner
    return None


def _rnn_first_layer_input_size(state: dict) -> int | None:
    for name, tensor in state.items():
        if name.endswith("weight_ih_l0") and isinstance(tensor, torch.Tensor):
            return int(tensor.shape[1])
    return None


def _apollo_rnn_base_dim(modalities: str, modality_proj_dim: int) -> int:
    if modalities == "at":
        return 2 * modality_proj_dim
    return int(apollo_constants.DIMS[modalities])


def _sync_use_pause_with_weights(
    ckpt: dict,
    modalities: str,
    modality_proj_dim: int,
    use_pause: bool,
) -> bool:
    """Согласовать use_pause с чекпоинтом, если в метаданных нет или они не совпадают с весами RNN."""
    sd = _state_dict_from_checkpoint(ckpt)
    if not sd:
        return use_pause
    rnn_in = _rnn_first_layer_input_size(sd)
    if rnn_in is None:
        return use_pause
    base = _apollo_rnn_base_dim(modalities, modality_proj_dim)
    if rnn_in == base + 1:
        if not use_pause:
            logger.info(
                "Чекпоинт: RNN input=%s → use_pause=True (в метаданных было False)",
                rnn_in,
            )
        return True
    if rnn_in == base:
        if use_pause:
            logger.info(
                "Чекпоинт: RNN input=%s → use_pause=False (в метаданных было True)",
                rnn_in,
            )
        return False
    raise ValueError(
        f"Размер входа RNN в чекпоинте ({rnn_in}) не согласуется с modalities={modalities!r} "
        f"и modality_proj_dim={modality_proj_dim}: ожидается {base} без паузы или {base + 1} с паузой."
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[Apollo, dict, Dict[str, Any]]:
    """Загрузка весов и гиперпараметров из ``results/.../model.pt`` (выход train.py)."""
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла чекпоинта: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cw = ckpt.get("class_weights")
    if cw is not None:
        cw = cw.to(device)

    ma = ckpt.get("model_args") or {}
    if "use_pause" in ckpt:
        use_pause = bool(ckpt["use_pause"])
    elif "use_pause" in ma:
        use_pause = bool(ma["use_pause"])
    else:
        use_pause = bool(apollo_constants.USE_PAUSE)

    modalities: str = (
        ma.get("modalities") or ckpt.get("modalities") or apollo_constants.MODALITIES
    )
    modality_feature_dim: int = int(
        ckpt.get(
            "modality_feature_dim",
            apollo_constants.DIMS[modalities],
        )
    )
    # совместимость с train.py: всегда есть modality_feature_dim
    dpb = int(ckpt.get("dialogues_per_batch", apollo_constants.DIALOGUES_PER_BATCH))

    if modalities not in apollo_constants.DIMS:
        raise ValueError(f"Неподдерживаемая модальность в чекпоинте: {modalities!r}")

    proj_dim = int(ma.get("modality_proj_dim", apollo_constants.MODALITY_PROJ_DIM))
    use_pause = _sync_use_pause_with_weights(ckpt, modalities, proj_dim, use_pause)

    model = Apollo(
        modalities=modalities,
        device=device,
        class_weights=cw,
        use_pause=use_pause,
    )
    model.to(device)
    if "best_state" in ckpt:
        model.load_state_dict(ckpt["best_state"], strict=False)
    elif "state_dict" in ckpt:
        inner = ckpt["state_dict"]
        if isinstance(inner, torch.nn.Module):
            model.load_state_dict(inner.state_dict(), strict=False)
        else:
            model.load_state_dict(inner, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    meta = {
        "modalities": modalities,
        "use_pause": use_pause,
        "modality_feature_dim": modality_feature_dim,
        "dialogues_per_batch": dpb,
    }
    return model, ma, meta


def _make_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        dataset_constants.SENTENCE_TRANSFORMER_MODEL,
        device=str(apollo_constants.DEVICE),
    )


def _text_and_audio_proxies(
    emb_model: SentenceTransformer,
    text: str,
    modalities: str,
    modality_feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Текстовый эмбеддинг 768. Для «a» (Wav2Vec) и «at» в поле ``audio_features`` подаётся
    тот же 768-вектор, что и при отсутствии записи микрофона (локальное демо).
    """
    t = emb_model.encode(text, convert_to_numpy=True)
    t = t.astype(np.float32, copy=False)
    if t.shape[-1] != dataset_constants.TEXT_EMBED_DIM:
        raise ValueError(
            f"Неверная размерность эмбеддинга: {t.shape}, ожидается {dataset_constants.TEXT_EMBED_DIM}"
        )
    if modalities == "a" and modality_feature_dim != dataset_constants.AUDIO_FEATURE_DIM:
        raise ValueError(
            f"Чекпоинт ожидает {modality_feature_dim} для «a», в данных: {dataset_constants.AUDIO_FEATURE_DIM}"
        )
    if modalities == "t" and modality_feature_dim != dataset_constants.TEXT_EMBED_DIM:
        raise ValueError(f"Чекпоинт ожидает {modality_feature_dim} для «t»")
    if modalities == "at" and modality_feature_dim != (
        dataset_constants.AUDIO_FEATURE_DIM + dataset_constants.TEXT_EMBED_DIM
    ):
        raise ValueError("Для at ожидается 1536 = 768+768 (см. DIMS[at])")
    # Dataset всегда вызывает _sample_audio_vec: для «t» подставляем нулевой «косяк», не участвующий в feat
    a_proxy = t.copy() if modalities in ("a", "at") else np.zeros(
        dataset_constants.AUDIO_FEATURE_DIM, dtype=np.float32
    )
    return t, a_proxy


def build_samples(
    emb_model: SentenceTransformer,
    turns: List[Dict[str, Any]],
    modalities: str,
    modality_feature_dim: int,
) -> List[Sample]:
    if not turns:
        raise ValueError("Пустой диалог")
    out: List[Sample] = []
    for i, u in enumerate(turns):
        text = (u.get("text") or "").strip()
        if not text:
            raise ValueError(f"Пустой текст в реплике {i}")
        sp = int(u["speaker"])
        t_np, a_proxy = _text_and_audio_proxies(
            emb_model, text, modalities, modality_feature_dim
        )
        s = Sample(
            utterance_id=f"u{i}",
            text=text,
            audio_path="",
            label=0,
            dialogue_id=0,
            start=float(i),
            end=float(i) + 1.0,
            embeddings=t_np,
            audio_features=a_proxy,
            speaker_name="",
            pause=0.0,
            speaker_id=sp,
        )
        out.append(s)
    return out


def predict_utterance_emotions(
    model: Apollo,
    emb_model: SentenceTransformer,
    turns: List[Dict[str, Any]],
    meta: Dict[str, Any],
    pause_mu: float,
    pause_std: float,
) -> List[Dict[str, Any]]:
    modalities = meta["modalities"]
    mfd = int(meta["modality_feature_dim"])
    dpb = int(meta.get("dialogues_per_batch", 1))
    use_pause = bool(meta["use_pause"])

    samples = build_samples(emb_model, turns, modalities, mfd)
    device = apollo_constants.DEVICE
    ds = Dataset(
        samples,
        dialogues_per_batch=dpb,
        modalities=modalities,
        modality_feature_dim=mfd,
        pause_mu=pause_mu,
        pause_std=pause_std,
        augment=False,
        use_pause=use_pause,
    )
    if len(ds) < 1:
        raise RuntimeError("Нет батчей в Dataset")
    data = ds[0]
    batch_to_device(data, device)
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        conf = probs.max(dim=-1).values.cpu().numpy().tolist()

    emap = dataset_constants.EMOTION_MAP
    idx_to_name = {v: k for k, v in emap.items()}
    names = [idx_to_name[int(p)] for p in pred]

    utterance_texts = data.get("utterance_texts") or []
    n = len(pred)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append(
            {
                "index": i,
                "text": utterance_texts[i] if i < len(utterance_texts) else turns[i][
                    "text"
                ],
                "speaker": int(turns[i]["speaker"]),
                "emotion": names[i],
                "emotion_id": int(pred[i]),
                "confidence": float(conf[i]) if i < len(conf) else None,
            }
        )
    return out
