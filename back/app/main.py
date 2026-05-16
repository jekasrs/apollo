"""
Eleos: локальный веб-интерфейс к модели Apollo — диалог → эмоции по репликам и граф.

Запуск (корень репозитория, PYTHONPATH указывает на каталог ``back/``):

  PYTHONPATH=back uvicorn app.main:app --host 127.0.0.1 --port 8765

или из каталога ``back/``: ``PYTHONPATH=. uvicorn app.main:app ...``

По умолчанию чекпоинт: results/apollo_meld_at_r01/model.pt
Другой файл: export APOLLO_CHECKPOINT=/path/to/model.pt
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BACK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACK_ROOT.parent
if str(BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(BACK_ROOT))

from dataset import SAMPLES_PKL
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as dataset_utils
from app.inference import load_checkpoint, _make_embedder, predict_utterance_emotions
from models.apollo.utils import constants as apollo_constants

STATIC = REPO_ROOT / "front"

_state: dict = {}

# Группы для «тональности» реплик (MELD / Apollo, 7 классов)
_EMOTION_POSITIVE = frozenset({"joy"})
_EMOTION_NEGATIVE = frozenset({"anger", "disgust", "fear", "sadness"})
_EMOTION_NEUTRAL_TONE = frozenset({"neutral", "surprise"})


def _tone_bucket(emotion: str) -> str:
    e = (emotion or "neutral").strip().lower()
    if e in _EMOTION_POSITIVE:
        return "positive"
    if e in _EMOTION_NEGATIVE:
        return "negative"
    if e in _EMOTION_NEUTRAL_TONE:
        return "neutral"
    return "neutral"


def _speaker_emotion_stats(
    utterance_rows: list[dict],
    num_speakers: int,
) -> list[dict]:
    """Доли позитивных / негативных / нейтральных реплик и разбивка по 7 эмоциям, по каждому спикеру."""
    tone_counts: dict[int, dict[str, int]] = {
        i: {"positive": 0, "negative": 0, "neutral": 0} for i in range(num_speakers)
    }
    em_counts: dict[int, dict[str, int]] = {
        i: defaultdict(int) for i in range(num_speakers)
    }
    totals = [0] * num_speakers

    for row in utterance_rows:
        sp = int(row["speaker"])
        if sp < 0 or sp >= num_speakers:
            continue
        em = str(row.get("emotion") or "neutral").strip().lower()
        totals[sp] += 1
        tone_counts[sp][_tone_bucket(em)] += 1
        if em not in dataset_constants.EMOTION_MAP:
            em = "neutral"
        em_counts[sp][em] += 1

    emotion_order = list(dataset_constants.EMOTION_MAP.keys())
    out: list[dict] = []
    for sp in range(num_speakers):
        t = totals[sp]
        tc = tone_counts[sp]
        if t > 0:
            tone_pct = {
                "positive": round(100.0 * tc["positive"] / t, 1),
                "negative": round(100.0 * tc["negative"] / t, 1),
                "neutral": round(100.0 * tc["neutral"] / t, 1),
            }
            em_pct = {
                e: round(100.0 * em_counts[sp][e] / t, 1) for e in emotion_order
            }
        else:
            tone_pct = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            em_pct = {e: 0.0 for e in emotion_order}
        out.append(
            {
                "speaker": sp,
                "utterances": t,
                "tone_percent": tone_pct,
                "emotion_percent": em_pct,
            }
        )
    return out


def _checkpoint_path(ck: str) -> Path:
    """Путь к чекпоинту: абсолютный как есть, относительный — от корня репозитория."""
    p = Path(ck)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _pause_stats():
    s0 = _state.get("pause_mu"), _state.get("pause_std")
    if s0[0] is not None and s0[1] is not None:
        return float(s0[0]), float(s0[1])
    if not SAMPLES_PKL.is_file():
        return 0.0, 1.0
    data = dataset_utils.load_pickle(SAMPLES_PKL)
    train = data.get("train") or []
    if not train:
        return 0.0, 1.0
    t0 = train[0]
    mu = getattr(t0, "pause_norm_mu", None)
    std = getattr(t0, "pause_norm_std", None)
    if mu is not None and std is not None:
        return float(mu), float(std)
    mu, std = dataset_utils.compute_pause_norm_stats(train)
    return float(mu), float(std)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _default_ck = REPO_ROOT / "results" / "apollo_meld_at_r01" / "model.pt"
    ck = os.environ.get("APOLLO_CHECKPOINT", str(_default_ck))
    ck_path = _checkpoint_path(ck)
    _state["checkpoint_path"] = str(ck_path)
    _state["root"] = str(REPO_ROOT)
    pm, ps = _pause_stats()
    _state["pause_mu"] = pm
    _state["pause_std"] = ps
    try:
        m, _ma, meta = load_checkpoint(ck_path, apollo_constants.DEVICE)
        _state["model"] = m
        _state["meta"] = meta
        _state["load_error"] = None
    except Exception as e:  # noqa: BLE001
        _state["model"] = None
        _state["meta"] = {}
        _state["load_error"] = f"{type(e).__name__}: {e}"
    _state["embedder"] = _make_embedder()
    yield
    _state.clear()


app = FastAPI(title="Eleos", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class TurnIn(BaseModel):
    text: str
    speaker: int = Field(ge=0, description="Индекс говорящего, 0 … num_speakers-1")


class AnalyzeBody(BaseModel):
    num_speakers: int = Field(ge=1, le=32)
    utterances: list[TurnIn]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {
        "ready": _state.get("model") is not None,
        "checkpoint": _state.get("checkpoint_path"),
        "device": str(apollo_constants.DEVICE),
        "load_error": _state.get("load_error"),
        "meta": _state.get("meta") or {},
    }


@app.post("/api/analyze")
def analyze(body: AnalyzeBody):
    m = _state.get("model")
    if m is None:
        raise HTTPException(
            503,
            detail=_state.get("load_error")
            or "Модель не загружена. Проверьте APOLLO_CHECKPOINT и наличие файла.",
        )
    uu = [u.model_dump() for u in body.utterances]
    if not uu:
        raise HTTPException(400, detail="Нет реплик")
    n = body.num_speakers
    for i, t in enumerate(uu):
        if t["speaker"] < 0 or t["speaker"] >= n:
            raise HTTPException(
                400,
                detail=f"Реплика {i}: speaker={t['speaker']} вне диапазона 0..{n - 1}",
            )
    emb = _state["embedder"]
    meta = _state["meta"] or {}
    pm, ps = _pause_stats()
    try:
        results = predict_utterance_emotions(
            m, emb, uu, meta, pm, ps
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail=str(e)) from e

    return {
        "num_speakers": n,
        "utterances": results,
        "speaker_stats": _speaker_emotion_stats(results, n),
    }
