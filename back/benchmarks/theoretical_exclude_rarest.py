#!/usr/bin/env python3
"""
Без обучения модели: оценка задачи «исключить K самых редких классов по частоте train».

Считает сохранённую долю реплик по сплитам, распределение оставшихся классов (метки 0..K-1),
majority baseline (то же, что всегда предсказывать самый частый класс train на test).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dataset import SAMPLES_PKL
from dataset.models.Sample import Sample
from dataset.preprocess.utils import constants as dataset_constants
from dataset.preprocess.utils import utils as dataset_utils


def _label_as_int(label) -> int:
    if hasattr(label, "item"):
        return int(label.item())
    return int(label)


def _train_label_counts(samples: list) -> Counter[int]:
    c: Counter[int] = Counter()
    for s in samples:
        c[_label_as_int(s.label)] += 1
    return c


def _rarest_class_indices(counts: Counter[int], k: int, full_map: dict[str, int]) -> list[int]:
    """Индексы K классов с наименьшим train-count (при равенстве — меньший числовой индекс)."""
    present = sorted(full_map.values())
    ranked = sorted(present, key=lambda idx: (counts.get(idx, 0), idx))
    return ranked[:k]


def _kept_classes(full_map: dict[str, int], excluded_indices: set[int]) -> list[tuple[str, int]]:
    out = [(name, idx) for name, idx in sorted(full_map.items(), key=lambda x: x[1])]
    return [(n, i) for n, i in out if i not in excluded_indices]


def _remap_kept(kept: list[tuple[str, int]]) -> tuple[dict[int, int], list[str]]:
    old_to_new = {old_i: j for j, (_n, old_i) in enumerate(kept)}
    names = [n for n, _ in kept]
    return old_to_new, names


def _filter_and_remap(samples: list, old_to_new: dict[int, int]) -> list[Sample]:
    out: list[Sample] = []
    for s in samples:
        li = _label_as_int(s.label)
        if li not in old_to_new:
            continue
        ni = old_to_new[li]
        lbl = s.label
        if hasattr(lbl, "new_tensor"):
            new_lbl = lbl.new_tensor(ni, dtype=lbl.dtype, device=lbl.device)
        else:
            new_lbl = ni
        out.append(
            Sample(
                s.utterance_id,
                s.text,
                s.audio_path,
                new_lbl,
                s.dialogue_id,
                s.start,
                s.end,
                s.embeddings,
                s.audio_features,
                s.speaker_name,
                pause=s.pause,
                speaker_id=s.speaker_id,
                pause_norm_mu=s.pause_norm_mu,
                pause_norm_std=s.pause_norm_std,
            )
        )
    return out


def _majority_accuracy(train_labels: list[int], test_labels: list[int]) -> tuple[int, float]:
    maj = Counter(train_labels).most_common(1)[0][0]
    correct = sum(1 for y in test_labels if y == maj)
    return maj, correct / max(len(test_labels), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-pkl",
        type=Path,
        default=None,
        help="samples.pkl (по умолчанию dataset.SAMPLES_PKL / APOLLO_SAMPLES_PKL)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Сколько самых редких классов убрать (по счётчику train).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Явный список имён через запятую (перекрывает авто-подбор по --k).",
    )
    args = parser.parse_args()

    pkl = args.samples_pkl.resolve() if args.samples_pkl is not None else SAMPLES_PKL
    data = dataset_utils.load_pickle(str(pkl))
    train = list(data["train"])
    dev = list(data["dev"])
    test = list(data["test"])

    emap = dataset_constants.EMOTION_MAP
    idx_to_name = {v: k for k, v in emap.items()}
    counts = _train_label_counts(train)

    if args.exclude.strip():
        names = [x.strip().lower() for x in args.exclude.split(",") if x.strip()]
        for n in names:
            if n not in emap:
                raise SystemExit(f"Неизвестная эмоция {n!r}. Допустимо: {sorted(emap)}")
        excluded = {emap[n] for n in names}
    else:
        rare_idx = _rarest_class_indices(counts, args.k, emap)
        excluded = set(rare_idx)
        names = [idx_to_name[i] for i in rare_idx]

    kept_pairs = _kept_classes(emap, excluded)
    old_to_new, kept_names = _remap_kept(kept_pairs)

    print(f"samples.pkl: {pkl}")
    print(f"Исключённые классы ({len(excluded)}): {names} → индексы {sorted(excluded)}")
    print(f"Остаётся классов: {len(kept_names)} — {kept_names}")
    print()
    print("Train counts (полная инвентаризация MELD по текущему pickle):")
    for idx in sorted(emap.values()):
        print(f"  {idx_to_name[idx]:10} {counts.get(idx, 0):6}")
    print()

    splits = {"train": train, "dev": dev, "test": test}
    filtered: dict[str, list] = {}
    for name, spl in splits.items():
        filtered[name] = _filter_and_remap(spl, old_to_new)

    for name in ("train", "dev", "test"):
        orig_n = len(splits[name])
        new_n = len(filtered[name])
        frac = new_n / orig_n if orig_n else 0.0
        print(f"{name}: реплик {new_n}/{orig_n} ({frac:.2%} сохранено)")

    print()
    for name in ("train", "dev", "test"):
        ctr = Counter(_label_as_int(s.label) for s in filtered[name])
        total = sum(ctr.values())
        parts = [f"{kept_names[j]}={ctr.get(j, 0)}" for j in range(len(kept_names))]
        print(f"{name} после фильтра ({total} реплик): " + ", ".join(parts))

    tr_y = [_label_as_int(s.label) for s in filtered["train"]]
    te_y = [_label_as_int(s.label) for s in filtered["test"]]
    maj_class, maj_acc = _majority_accuracy(tr_y, te_y)
    print()
    print(
        "Majority baseline на filtered test "
        f"(всегда класс train-{maj_class} = «{kept_names[maj_class]}»): "
        f"accuracy = {maj_acc:.4f}"
    )
    print(
        "(Для сравнения: случайное угадывание по частотам train даёт ту же долю, что и стратифицированный dummy; "
        "weighted F1 зависит от баланса классов на test.)"
    )


if __name__ == "__main__":
    main()
