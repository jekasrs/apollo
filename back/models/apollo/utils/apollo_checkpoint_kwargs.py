"""Поля Apollo, сериализуемые в checkpoint model_args (train → eval / inference)."""


def optional_apollo_kwargs_from_ma(ma: dict) -> dict:
    """Значения по умолчанию совместимы со старыми чекпоинтами без этих ключей."""
    return {
        "use_speaker_embedding": bool(ma.get("use_speaker_embedding", False)),
        "speaker_emb_dim": int(ma.get("speaker_emb_dim", 32)),
        "max_local_speakers": int(ma.get("max_local_speakers", 24)),
        "graph_similarity_topk": int(ma.get("graph_similarity_topk", 0)),
        "graph_similarity_min_cos": float(ma.get("graph_similarity_min_cos", 0.35)),
        "emotion_shift_loss_weight": float(ma.get("emotion_shift_loss_weight", 0.0)),
        "graph_wp": int(ma.get("graph_wp", 10)),
        "graph_wf": int(ma.get("graph_wf", 10)),
    }
