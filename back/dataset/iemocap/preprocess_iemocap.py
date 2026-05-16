"""
IEMOCAP → тот же формат, что MELD (Sample + split), 7 эмоций MELD.
Требуется **локальная** распаковка IEMOCAP (LDC, не скачиваем).

Пример (из корня репозитория):
  PYTHONPATH=. python3 dataset/iemocap/preprocess_iemocap.py \\
    --iemocap-root /path/to/IEMOCAP_full_release \\
    --out dataset/preprocess/samples/iemocap_meld7.pkl

Опционально CSV с колонками utt_id + text (транскрипты реплик):
  --transcript-csv /path/utt_text.csv

Дальше Apollo без изменений архитектуры:
  PYTHONPATH=. python3 models/apollo/trainings/train.py --modalities at --use-pause --samples-pkl dataset/preprocess/samples/iemocap_meld7.pkl
  # дообучение с весов MELD (опционально):
  # ... --from-checkpoint results/apollo_meld_at_r01/model.pt
  PYTHONPATH=. python3 models/apollo/trainings/eval.py --samples-pkl dataset/preprocess/samples/iemocap_meld7.pkl --checkpoint results/apollo_meld_at_r01/model.pt
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dataset.models.Sample import Sample
from dataset.iemocap.parse_tree import IERow, _dialogue_id_from_utt, collect_iemocap_rows
from dataset.preprocess.preprocess import _build_encoders
from dataset.preprocess.utils import utils as u
from dataset.preprocess.utils import constants as c

log = logging.getLogger(__name__)
BATCH = c.WAV2VEC_BATCH_SIZE


def _order_rows_for_samples(rows: list[IERow]) -> list[IERow]:
    by: dict[str, list[IERow]] = defaultdict(list)
    for r in rows:
        by[_dialogue_id_from_utt(r.utt_id)].append(r)
    for k in by:
        by[k].sort(key=lambda x: (x.start, x.end))
    out: list[IERow] = []
    for k in sorted(by.keys()):
        out.extend(by[k])
    return out


def _speaker_from_utt(utt_id: str) -> str:
    import re
    m = re.search(r"([FM])(\d+)$", utt_id)
    if m:
        return m.group(1)
    return "U"


def build_iemocap_pickle(
    iemocap_root: Path,
    out_path: Path,
    transcript_csv: Path | None,
    test_size: float,
    dev_size: float,
    seed: int,
) -> None:
    text_model, audio_model = _build_encoders()
    rows = collect_iemocap_rows(iemocap_root, transcript_csv=transcript_csv)
    if not rows:
        raise SystemExit("Нет валидных (wav+эмоция) сегментов. Проверьте --iemocap-root.")
    rows = _order_rows_for_samples(rows)
    by_d: dict[str, list[IERow]] = defaultdict(list)
    for r in rows:
        by_d[_dialogue_id_from_utt(r.utt_id)].append(r)
    for k in by_d:
        by_d[k].sort(key=lambda x: (x.start, x.end))
    # плоский список (порядок: диалоги по имени) с локальным utterance_id
    flat: list[tuple[IERow, int, str]] = []
    for dkey in sorted(by_d.keys()):
        for j, r in enumerate(by_d[dkey]):
            flat.append((r, j, dkey))
    log.info("Всего реплик к кодированию: %d, диалогов: %d", len(flat), len(by_d))

    samples: list[Sample] = []
    pending: list[dict[str, Any]] = []

    def flush():
        nonlocal pending
        if not pending:
            return
        texts = [p["text"] for p in pending]
        embs = text_model.encode_batch(texts)  # type: ignore[union-attr]
        aids = [p["audio"] for p in pending]
        feats = audio_model.encode_batch(aids)
        for p, emb, aud in zip(pending, embs, feats):
            e = (
                emb.tolist() if hasattr(emb, "tolist") else (list(emb) if isinstance(emb, (list, tuple)) else list(emb))
            )
            samples.append(
                Sample(
                    utterance_id=p["utterance_id"],
                    text=p["text"],
                    audio_path=p["audio_path"],
                    label=p["label"],
                    dialogue_id=p["dialogue_id"],
                    start=p["start"],
                    end=p["end"],
                    embeddings=e,
                    audio_features=aud,
                    speaker_name=p["speaker"],
                )
            )
        pending = []

    for r, uidx, did in tqdm(flat, desc="encode"):
        p = r.wav_path
        assert p is not None
        y, _sr = u.load_audio_segment(p)
        y = u.normalize_audio(y)
        t = u.clean_text(r.text, remove_stopwords=False)
        pending.append(
            {
                "utterance_id": uidx,
                "text": t,
                "audio_path": str(p),
                "label": r.meld_label,
                "dialogue_id": did,
                "start": r.start,
                "end": r.end,
                "audio": y,
                "speaker": _speaker_from_utt(r.utt_id),
            }
        )
        if len(pending) >= BATCH:
            flush()
    flush()

    u.assign_pause_until_next_in_dialogues(samples)
    train, dev, test = u.split_samples_by_dialogue(samples, test_size=test_size, dev_size=dev_size, random_state=seed)
    p_mu, p_std = u.compute_pause_norm_stats(train)
    p_mu, p_std = float(p_mu), float(p_std)
    for s in train + dev + test:
        s.pause_norm_mu = p_mu
        s.pause_norm_std = p_std
    out = {"train": train, "dev": dev, "test": test}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u.save_pickle(out, str(out_path))
    log.info("Сохранено: %s (train %d, dev %d, test %d)", out_path, len(train), len(dev), len(test))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="IEMOCAP → samples.pkl (7 классов MELD)")
    ap.add_argument("--iemocap-root", type=Path, required=True, help="Корень распакованного IEMOCAP (Session* внутри)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/preprocess/samples/iemocap_meld7.pkl"),
        help="Выходной pickle",
    )
    ap.add_argument("--transcript-csv", type=Path, default=None, help="Опц.: CSV с utt_id и text")
    ap.add_argument("--test-size", type=float, default=c.TEST_SIZE)
    ap.add_argument("--dev-size", type=float, default=c.DEV_SIZE)
    ap.add_argument("--seed", type=int, default=c.RANDOM_STATE)
    args = ap.parse_args()
    build_iemocap_pickle(
        args.iemocap_root,
        args.out,
        args.transcript_csv,
        test_size=args.test_size,
        dev_size=args.dev_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
