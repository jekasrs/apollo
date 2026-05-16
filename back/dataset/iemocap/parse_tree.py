"""
Сканирование корня IEMOCAP (Session*/…): EmoEvaluation-файлы + wav в sentences/wav.
Без скачивания: пользователь кладёт распакованный LDC-релиз.

Формат строк (типичный):
  [6.2901 - 8.2357]\\tSes01F_impro01_F000\\tneu\\t[2.5,2.5,2.5]
или:
  [6.29 - 8.24]  Ses01F_impro01_F000  neu
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path

from dataset.iemocap.labels import iemocap_abbrev_to_meld_label

log = logging.getLogger(__name__)


@dataclass
class IERow:
    utt_id: str
    start: float
    end: float
    emo_raw: str
    meld_label: int
    wav_path: Path | None = None
    text: str = ""


def _find_wav(iemocap_root: Path, utt_id: str) -> Path | None:
    """Ses01F_impro01_F000 → …/Session*/sentences/wav/Ses01F_impro01_F000.wav"""
    w = f"{utt_id}.wav"
    for p in iemocap_root.rglob(w):
        if p.is_file() and "sentences" in p.as_posix() and "wav" in p.as_posix():
            return p
    for p in iemocap_root.rglob(w):
        if p.is_file():
            return p
    return None


def _load_optional_transcript_map(csv_path: Path) -> dict[str, str]:
    import csv

    m: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        tcol = ucol = None
        for k in (r.fieldnames or []):
            kl = (k or "").lower().strip()
            if "text" in kl or "transcript" in kl or "utter" in kl:
                tcol = k
            if "id" in kl or "utt" in kl or "file" in kl:
                ucol = k
        if not tcol or not ucol:
            raise SystemExit("CSV: нужны колонки с id реплики (utt_id) и текстом")
        f.seek(0)
        f.readline()
        r = csv.DictReader(f)
        for row in r:
            uid = (row.get(ucol) or "").strip()
            if uid.endswith(".wav"):
                uid = uid[: -4]
            t = (row.get(tcol) or "").strip()
            if uid:
                m[uid] = t
    return m


def _try_per_utterance_txt(iemocap_root: Path, utt_id: str) -> str:
    w = f"{utt_id}.txt"
    for p in iemocap_root.rglob(w):
        if p.is_file() and "txt" in p.as_posix():
            try:
                return p.read_text(encoding="utf-8", errors="replace").strip()[:2000]
            except OSError:
                continue
    return ""


def _dialogue_id_from_utt(utt_id: str) -> str:
    m = re.match(r"^(.+)_([FM][\d]+)$", utt_id)
    if m:
        return m.group(1)
    parts = utt_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return utt_id


def _parse_emoeval_file(path: Path) -> list[tuple[str, float, float, str]]:
    """Возвращает (utt_id, start, end, emo_abbrev) по строкам файла."""
    out: list[tuple[str, float, float, str]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m1 = re.match(
            r"^\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\S+?)(?:\s+\[.*\])?\s*$", line
        )
        if m1:
            t0, t1, uid, emo = float(m1.group(1)), float(m1.group(2)), m1.group(3), m1.group(4)
            if uid.endswith(".wav"):
                uid = uid[:-4]
            out.append((uid, t0, t1, emo))
            continue
        m2 = re.match(
            r"^(\S+)\s+\[([\d.]+),([\d.]+)\]\s+(\S+)", line
        )
        if m2:
            uid, t0, t1, emo = m2.group(1), float(m2.group(2)), float(m2.group(3)), m2.group(4)
            if uid.endswith(".wav"):
                uid = uid[:-4]
            out.append((uid, t0, t1, emo))
    return out


def collect_iemocap_rows(
    iemocap_root: Path,
    transcript_csv: Path | None = None,
) -> list[IERow]:
    root = iemocap_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Нет каталога IEMOCAP: {root}")

    tmap: dict[str, str] = {}
    if transcript_csv and transcript_csv.is_file():
        tmap = _load_optional_transcript_map(transcript_csv)
        log.info("Загружено %d транскриптов из %s", len(tmap), transcript_csv)

    eval_files: list[Path] = []
    for pat in (
        "Session*/dialog/EmoEvaluation/*.txt",
        "*/dialog/EmoEvaluation/*.txt",
        "Session*/**/*EmoEvaluation*.txt",
    ):
        eval_files.extend(root.glob(pat))
    eval_files = sorted(set([p for p in eval_files if p.is_file()]))
    if not eval_files:
        eval_files = sorted(root.rglob("*EmoEvaluation.txt"))
    if not eval_files:
        raise SystemExit(
            f"Не найдены *EmoEvaluation*.txt под {root}. "
            f"Проверьте путь к распакованному IEMOCAP (Session1/dialog/EmoEvaluation/...)."
        )
    log.info("EmoEvaluation файлов: %d", len(eval_files))

    seen: set[str] = set()
    raw_rows: list[tuple[str, float, float, str]] = []
    for fp in eval_files:
        for row in _parse_emoeval_file(fp):
            if row[0] in seen:
                continue
            seen.add(row[0])
            raw_rows.append(row)

    log.info("Уникальных реплик по разметке: %d", len(raw_rows))
    out: list[IERow] = []
    for utt_id, t0, t1, emo in raw_rows:
        lab = iemocap_abbrev_to_meld_label(emo)
        if lab is None:
            continue
        wav = _find_wav(root, utt_id)
        text = tmap.get(utt_id, "") or tmap.get(utt_id + ".wav", "")
        if not text:
            text = _try_per_utterance_txt(root, utt_id)
        if not text:
            # минимальная клюжева, чтобы 768d текст не деградировал в пустоту; лучше дать --transcript-csv
            text = f"speech emotion {emo}"
        out.append(
            IERow(
                utt_id=utt_id,
                start=t0,
                end=t1,
                emo_raw=emo,
                meld_label=lab,
                wav_path=wav,
                text=text,
            )
        )
    nskipped = sum(1 for (_, _, _, e) in raw_rows if iemocap_abbrev_to_meld_label(e) is None)
    if nskipped:
        log.info("Пропущено (xxx/oth/нецелевые эмоции): %d", nskipped)
    nmiss = len([r for r in out if r.wav_path is None])
    if nmiss:
        log.warning("Нет .wav на диске для %d реплик (будут пропущены).", nmiss)
    return [r for r in out if r.wav_path is not None]
