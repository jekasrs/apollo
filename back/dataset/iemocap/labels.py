"""Маппинг 3-буквенных (и вариаций) меток IEMOCAP в индексы MELD (EMOTION_MAP)."""
from __future__ import annotations

# neu, ang, ... из EmoEvaluation; oth/xxx — непригодны для 7-class
IEMOCAP_TO_MELD = {
    "neu": "neutral",
    "neutral": "neutral",
    "hap": "joy",
    "happy": "joy",
    "xxx": None,
    "oth": None,
    "sad": "sadness",
    "sadness": "sadness",
    "ang": "anger",
    "anger": "anger",
    "fru": "anger",  # frustration → anger
    "exc": "joy",  # excited
    "fea": "fear",
    "fear": "fear",
    "sur": "surprise",
    "surprise": "surprise",
    "dis": "disgust",
    "disgust": "disgust",
    "lau": "joy",  # laughter → joy (звуковой класс, редк.)
    "cal": "neutral",  # calm
    "n/a": None,
    "nan": None,
}


def iemocap_abbrev_to_meld_label(abbrev: str) -> int | None:
    if not abbrev or not str(abbrev).strip():
        return None
    a = str(abbrev).lower().strip()
    a = a.replace(".", "")
    if len(a) > 3 and a[:3] in ("neu", "hap", "sad", "ang", "exc", "fru", "sur", "dis", "fea", "lau", "oth", "cal"):
        a = a[:3]
    name = IEMOCAP_TO_MELD.get(a) or IEMOCAP_TO_MELD.get(a[:3] if len(a) >= 3 else a)
    if name is None:
        return None
    from dataset.preprocess.utils import constants as c

    return c.EMOTION_MAP.get(name)  # type: ignore[return-value]
