"""
Evaluate a trained Apollo checkpoint on the test split.

Usage (from project root):
  PYTHONPATH=. python3 models/apollo/trainings/eval.py
  PYTHONPATH=. python3 models/apollo/trainings/eval.py --checkpoint results/apollo_meld_at_r01/model.pt
  # Подобрать сдвиг logit'ов на dev, затем отчёт на test (только dev участвует в подборе):
  PYTHONPATH=. python3 models/apollo/trainings/eval.py --tune-bias
  PYTHONPATH=. python3 models/apollo/trainings/eval.py --checkpoint x/model.pt \\
    --tune-bias --tune-bias-metric weighted_f1
  PYTHONPATH=. python3 models/apollo/trainings/eval.py \\
    --ensemble-checkpoints results/a/model.pt results/b/model.pt --tune-bias --tune-bias-metric weighted_f1
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from dataset import SAMPLES_PKL
from dataset.models.Dataset import Dataset
from dataset.preprocess.utils import utils as dataset_utils
from dataset.preprocess.utils import constants as dataset_constants
from models.apollo.utils import constants as arguments_and_constants

from models.apollo.Apollo import Apollo
from models.apollo.utils.apollo_checkpoint_kwargs import optional_apollo_kwargs_from_ma
from models.apollo.utils.functions import batch_to_device
from models.apollo.utils.repo_paths import default_run_results_dir

logging.basicConfig(level=logging.INFO)

_DEFAULT_CHECKPOINT = default_run_results_dir() / "model.pt"


def evaluate_dataset(
    model,
    dataset,
    device,
    label_to_idx,
    desc="eval",
    print_report=False,
    logit_bias: np.ndarray | None = None,
):
    model.eval()
    total_loss = 0.0
    golds = []
    preds = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc):
            data = dataset[idx]
            golds.append(data["label_tensor"])
            batch_to_device(data, device)
            y_hat = model(data)
            preds.append(y_hat.detach().cpu())
            total_loss += model.get_loss(data).item()

    golds = torch.cat(golds, dim=0).numpy()
    logits = torch.cat(preds, dim=0).numpy()
    if logit_bias is not None:
        b = np.asarray(logit_bias, dtype=np.float64)
        if b.shape != (logits.shape[1],):
            raise ValueError(
                f"logit_bias must have shape ({logits.shape[1]},), got {b.shape}"
            )
        pred_labels = np.argmax(logits + b.reshape(1, -1), axis=1)
    else:
        pred_labels = np.argmax(logits, axis=1)
    weighted_f1 = metrics.f1_score(golds, pred_labels, average="weighted", zero_division=0)
    macro_f1 = metrics.f1_score(golds, pred_labels, average="macro", zero_division=0)
    acc = metrics.accuracy_score(golds, pred_labels)
    n_batches = len(dataset)
    mean_loss = total_loss / n_batches if n_batches else 0.0

    per_class = metrics.f1_score(
        golds,
        pred_labels,
        average=None,
        labels=np.arange(len(label_to_idx)),
        zero_division=0,
    )
    per_class_f1 = dict(zip(label_to_idx.keys(), per_class))

    if print_report:
        _labels = np.arange(len(label_to_idx))
        print(
            metrics.classification_report(
                golds,
                pred_labels,
                labels=_labels,
                target_names=list(label_to_idx.keys()),
                digits=4,
                zero_division=0,
            )
        )
        print(
            f"Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}  Weighted F1: {weighted_f1:.4f}  "
            f"Loss (sum/n_batches): {mean_loss:.4f}"
        )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_loss": mean_loss,
        "per_class_f1": per_class_f1,
    }


def sklearn_metrics_numpy(
    golds: np.ndarray,
    logits: np.ndarray,
    label_to_idx: dict,
    logit_bias: np.ndarray | None = None,
    print_report: bool = True,
) -> dict:
    """Отчёт по сохранённым логитам (ансамбль / постобработка без повторного inference)."""
    g = golds.astype(np.int64).ravel()
    L = logits.astype(np.float64)
    if logit_bias is not None:
        b = np.asarray(logit_bias, dtype=np.float64)
        pred_labels = np.argmax(L + b.reshape(1, -1), axis=1)
    else:
        pred_labels = np.argmax(L, axis=1)
    weighted_f1 = metrics.f1_score(g, pred_labels, average="weighted", zero_division=0)
    macro_f1 = metrics.f1_score(g, pred_labels, average="macro", zero_division=0)
    acc = metrics.accuracy_score(g, pred_labels)
    per_class = metrics.f1_score(
        g,
        pred_labels,
        average=None,
        labels=np.arange(len(label_to_idx)),
        zero_division=0,
    )
    per_class_f1 = dict(zip(label_to_idx.keys(), per_class))

    if print_report:
        _labels = np.arange(len(label_to_idx))
        print(
            metrics.classification_report(
                g,
                pred_labels,
                labels=_labels,
                target_names=list(label_to_idx.keys()),
                digits=4,
                zero_division=0,
            )
        )
        print(f"Accuracy: {acc:.4f}  Macro F1: {macro_f1:.4f}  Weighted F1: {weighted_f1:.4f}")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
    }


def collect_logits_and_golds(
    model, dataset, device, desc: str = "collect"
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    golds, preds = [], []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc):
            data = dataset[idx]
            golds.append(data["label_tensor"])
            batch_to_device(data, device)
            y_hat = model(data)
            preds.append(y_hat.detach().cpu())
    g = torch.cat(golds, dim=0).numpy()
    l = torch.cat(preds, dim=0).numpy()
    return g, l


def fit_logit_bias(
    logits: np.ndarray,
    gold: np.ndarray,
    n_classes: int,
    n_trials: int = 80000,
    seed: int = 0,
    objective: str = "accuracy",
    low: float = -3.0,
    high: float = 3.0,
) -> tuple[np.ndarray, float, float]:
    """
    Сдвиг logit'ов на dev случайным поиском без утечки test.

    objective: accuracy | weighted_f1 | macro_f1 — что максимизируем при подборе.
    Возвращает (best_bias, baseline_metric, best_metric после подбора bias).
    """
    rng = np.random.default_rng(seed)
    y = gold.astype(np.int64).ravel()
    logits = np.asarray(logits, dtype=np.float64)
    pred_base = logits.argmax(axis=1)

    def _score(labels: np.ndarray, pred: np.ndarray) -> float:
        if objective == "accuracy":
            return float((pred == labels).mean())
        if objective == "weighted_f1":
            return float(metrics.f1_score(labels, pred, average="weighted", zero_division=0))
        if objective == "macro_f1":
            return float(metrics.f1_score(labels, pred, average="macro", zero_division=0))
        raise ValueError(f"unknown objective: {objective!r}")

    baseline_scr = _score(y, pred_base)
    best = float(baseline_scr)
    best_b = np.zeros(n_classes, dtype=np.float64)

    for _ in range(max(1, n_trials)):
        b = rng.uniform(low, high, size=n_classes)
        pred = (logits + b.reshape(1, -1)).argmax(axis=1)
        sc = _score(y, pred)
        if sc > best + 1e-12:
            best, best_b = sc, b.copy()
    return best_b, float(baseline_scr), float(best)


def load_train_checkpoint(bundle: dict | Path, map_location) -> tuple[Apollo, dict]:
    """Apollo + полный ckpt-дикт для train-сохранения (ключ best_state)."""
    ckpt = (
        bundle
        if isinstance(bundle, dict)
        else torch.load(bundle, map_location=map_location, weights_only=False)
    )
    cw = ckpt.get("class_weights")
    if cw is not None:
        cw = cw.to(map_location)

    ma = ckpt.get("model_args") or {}
    if "use_pause" in ckpt:
        use_pause = bool(ckpt["use_pause"])
    elif "use_pause" in ma:
        use_pause = bool(ma["use_pause"])
    else:
        use_pause = True

    modalities = ma.get("modalities") or arguments_and_constants.MODALITIES
    gnn_edge_mode = ma.get("gnn_edge_mode", "heterogeneous")
    use_heterogeneous_gnn = gnn_edge_mode != "homogeneous"
    use_f = ma.get("use_focal_loss")
    if use_f is None:
        use_f = arguments_and_constants.USE_FOCAL_LOSS
    fg = ma.get("focal_gamma", arguments_and_constants.FOCAL_GAMMA)
    ls = ma.get("label_smoothing", arguments_and_constants.LABEL_SMOOTHING)
    fg = float(fg) if isinstance(fg, (float, int)) else arguments_and_constants.FOCAL_GAMMA
    ls = float(ls) if isinstance(ls, (float, int)) else arguments_and_constants.LABEL_SMOOTHING

    model = Apollo(
        modalities=modalities,
        device=map_location,
        class_weights=cw,
        use_pause=use_pause,
        focal_gamma=fg,
        use_focal=use_f,
        label_smoothing=ls,
        use_heterogeneous_gnn=use_heterogeneous_gnn,
        **optional_apollo_kwargs_from_ma(ma),
    ).to(map_location)

    met = {}
    if "best_state" in ckpt and isinstance(ckpt["best_state"], dict):
        inc = model.load_state_dict(ckpt["best_state"], strict=False)
        if getattr(inc, "missing_keys", None) or getattr(inc, "unexpected_keys", None):
            logging.warning(
                "Частичная загрузка весов (strict=False): missing=%s unexpected=%s",
                len(inc.missing_keys),
                len(inc.unexpected_keys),
            )
        met = {k: v for k, v in ckpt.items() if k not in ("best_state", "state_dict")}
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        if isinstance(sd, torch.nn.Module):
            inc = model.load_state_dict(sd.state_dict(), strict=False)
        else:
            inc = model.load_state_dict(sd, strict=False)
        if getattr(inc, "missing_keys", None) or getattr(inc, "unexpected_keys", None):
            logging.warning(
                "Частичная загрузка весов (strict=False): missing=%s unexpected=%s",
                len(inc.missing_keys),
                len(inc.unexpected_keys),
            )
        met = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        raise RuntimeError("В чекпоинте нет best_state / state_dict")
    return model, {**met, "__raw": ckpt, "modalities": modalities, "use_pause": use_pause, "gnn_edge_mode": gnn_edge_mode}


def _dataset_options_from_ckpt(ckpt: dict) -> tuple[bool, str, int, int]:
    """use_pause, modalities, dialogues_per_batch, modality_feature_dim."""
    ma = ckpt.get("model_args") or {}
    if "use_pause" in ckpt:
        use_pause = bool(ckpt["use_pause"])
    elif "use_pause" in ma:
        use_pause = bool(ma["use_pause"])
    else:
        use_pause = True
    modalities = ma.get("modalities") or arguments_and_constants.MODALITIES
    dpb = int(ckpt.get("dialogues_per_batch", arguments_and_constants.DIALOGUES_PER_BATCH))
    mfd = ckpt.get("modality_feature_dim")
    if mfd is None:
        mfd = arguments_and_constants.DIMS[modalities]
    return use_pause, modalities, dpb, int(mfd)


def _assert_ckpt_geometry_equal(ck_a: dict, ck_b: dict, label_a: str, label_b: str) -> None:
    ga, gb = _dataset_options_from_ckpt(ck_a), _dataset_options_from_ckpt(ck_b)
    if ga != gb:
        raise SystemExit(
            f"Ансамбль несовместим: геометрия данных {ga} ≠ {gb} ({label_a} vs {label_b})"
        )


def run_eval_ensemble(
    ckpt_paths: list[Path],
    pkl: Path,
    *,
    tune_bias: bool,
    tune_bias_metric: str,
    bias_trials: int,
    export_metrics_json: Path | None,
) -> None:
    device = arguments_and_constants.DEVICE
    resolved = [p.resolve() for p in ckpt_paths]
    bundles = [
        torch.load(p, map_location=device, weights_only=False) for p in resolved
    ]
    for i in range(1, len(bundles)):
        _assert_ckpt_geometry_equal(
            bundles[0], bundles[i], str(resolved[0]), str(resolved[i])
        )

    use_pause, modalities, dpb, modality_feature_dim = _dataset_options_from_ckpt(bundles[0])
    logging.info(
        "Ансамбль из %d моделей (%s pause=%s dpb=%d mfdim=%d)",
        len(resolved),
        modalities,
        use_pause,
        dpb,
        modality_feature_dim,
    )

    data = dataset_utils.load_pickle(f"{pkl.resolve()}")
    train_samples = data["train"]
    s0 = train_samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        pause_mu, pause_std = float(mu), float(std)
    else:
        pause_mu, pause_std = dataset_utils.compute_pause_norm_stats(train_samples)
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        logging.warning("pickle: нет pause_norm_mu/std — пересчёт по train")

    ds_kw = dict(
        dialogues_per_batch=dpb,
        modalities=modalities,
        modality_feature_dim=modality_feature_dim,
        pause_mu=pause_mu,
        pause_std=pause_std,
        augment=False,
        use_pause=use_pause,
    )
    dev_ds = Dataset(data["dev"], **ds_kw)
    test_ds = Dataset(data["test"], **ds_kw)

    dev_logits_list: list[np.ndarray] = []
    test_logits_list: list[np.ndarray] = []
    gold_dev_ref = gold_test_ref = None

    for i, bundle in enumerate(bundles):
        model, _ = load_train_checkpoint(bundle, device)
        gd, ld = collect_logits_and_golds(model, dev_ds, device, desc=f"ensemble dev [{i + 1}]")
        gt, lt = collect_logits_and_golds(model, test_ds, device, desc=f"ensemble test [{i + 1}]")
        if gold_dev_ref is None:
            gold_dev_ref = gd
            gold_test_ref = gt
        else:
            if not np.array_equal(gold_dev_ref, gd) or not np.array_equal(gold_test_ref, gt):
                raise SystemExit("Разный порядок примеров в батчах — ансамбль отменён")
        dev_logits_list.append(ld)
        test_logits_list.append(lt)
        del model

    assert gold_dev_ref is not None and gold_test_ref is not None
    Ld = np.mean(np.stack(dev_logits_list, axis=0), axis=0)
    Lt = np.mean(np.stack(test_logits_list, axis=0), axis=0)

    logit_bias = None
    if tune_bias:
        b, base_obj, tuned_obj = fit_logit_bias(
            Ld,
            gold_dev_ref,
            n_classes=Ld.shape[1],
            n_trials=bias_trials,
            seed=0,
            objective=tune_bias_metric,
        )
        logit_bias = b
        naive_d = sklearn_metrics_numpy(
            gold_dev_ref, Ld, dataset_constants.EMOTION_MAP, None, False
        )
        tune_d = sklearn_metrics_numpy(
            gold_dev_ref, Ld, dataset_constants.EMOTION_MAP, b, False
        )
        logging.info(
            "ensemble dev tune %s: obj %.5f→%.5f | wF1 %.4f→%.4f macro %.4f→%.4f | bias(head)≈ %s…",
            tune_bias_metric,
            base_obj,
            tuned_obj,
            naive_d["weighted_f1"],
            tune_d["weighted_f1"],
            naive_d["macro_f1"],
            tune_d["macro_f1"],
            np.round(logit_bias, 4)[: min(7, len(logit_bias))].tolist(),
        )

    logging.info("=== Test после усреднения логитов ===")
    pr = export_metrics_json is None
    test_metrics = sklearn_metrics_numpy(
        gold_test_ref,
        Lt,
        dataset_constants.EMOTION_MAP,
        logit_bias,
        print_report=pr,
    )

    if export_metrics_json is not None:
        out = {
            "ensemble_checkpoints": [str(p) for p in resolved],
            "samples_pkl": str(pkl.resolve()),
            "tune_bias_metric": tune_bias_metric if tune_bias else None,
            "accuracy": test_metrics["accuracy"],
            "weighted_f1": test_metrics["weighted_f1"],
            "macro_f1": test_metrics["macro_f1"],
            "per_emotion_f1": {k: float(v) for k, v in test_metrics["per_class_f1"].items()},
        }
        export_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        export_metrics_json.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logging.info("Метрики записаны в %s", export_metrics_json)


def _load_checkpoint_weights(model: Apollo, path: Path, map_location) -> dict:
    """Загрузка весов без повторной сборки модели (как прежний путь одного чекпоинта)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "best_state" and k != "state_dict"}
    if "best_state" in ckpt:
        model.load_state_dict(ckpt["best_state"], strict=False)
        return meta
    if "state_dict" in ckpt:
        inner = ckpt["state_dict"]
        if isinstance(inner, torch.nn.Module):
            model.load_state_dict(inner.state_dict(), strict=False)
        else:
            model.load_state_dict(inner, strict=False)
        return meta
    model.load_state_dict(ckpt, strict=False)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Оценка Apollo по сохранённому чекпоинту")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help=f"Чекпоинт от train.py (по умолчанию {_DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--tune-bias",
        action="store_true",
        help="Подобрать сдвиг logit'ов на dev, затем отчёт на test (без утечки: test не используется в подборе).",
    )
    parser.add_argument(
        "--samples-pkl",
        type=Path,
        default=None,
        help="Путь к samples.pkl (как при train, для test/dev).",
    )
    parser.add_argument(
        "--export-metrics-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Сохранить метрики test (accuracy, weighted_f1, f1 по классам) в JSON без лишних отчётов.",
    )
    parser.add_argument(
        "--tune-bias-metric",
        choices=["accuracy", "weighted_f1", "macro_f1"],
        default="accuracy",
        help="При --tune-bias: что максимизируем случайным поиском сдвига на dev.",
    )
    parser.add_argument(
        "--bias-tune-trials",
        type=int,
        default=120_000,
        help="Число проб подбора сдвига логитов (--tune-bias); больше — точнее и дольше.",
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        type=Path,
        default=None,
        metavar="PT",
        help=(
            "Усреднить логиты нескольких model.pt при одинаковой геометрии данных; "
            "с --checkpoint не используется."
        ),
    )
    args = parser.parse_args()
    pkl = args.samples_pkl.resolve() if args.samples_pkl is not None else SAMPLES_PKL
    if args.ensemble_checkpoints:
        run_eval_ensemble(
            args.ensemble_checkpoints,
            pkl,
            tune_bias=args.tune_bias,
            tune_bias_metric=args.tune_bias_metric,
            bias_trials=args.bias_tune_trials,
            export_metrics_json=args.export_metrics_json,
        )
        return

    ckpt_path = args.checkpoint

    ckpt = torch.load(
        ckpt_path, map_location=arguments_and_constants.DEVICE, weights_only=False
    )
    cw = ckpt.get("class_weights")
    if cw is not None:
        cw = cw.to(arguments_and_constants.DEVICE)

    ma = ckpt.get("model_args") or {}
    if "use_pause" in ckpt:
        use_pause = bool(ckpt["use_pause"])
    elif "use_pause" in ma:
        use_pause = bool(ma["use_pause"])
    else:
        use_pause = True

    modalities = ma.get("modalities") or arguments_and_constants.MODALITIES
    gnn_edge_mode = ma.get("gnn_edge_mode", "heterogeneous")
    use_heterogeneous_gnn = gnn_edge_mode != "homogeneous"
    use_f = ma.get("use_focal_loss")
    if use_f is None:
        use_f = arguments_and_constants.USE_FOCAL_LOSS
    fg = ma.get("focal_gamma", arguments_and_constants.FOCAL_GAMMA)
    ls = ma.get("label_smoothing", arguments_and_constants.LABEL_SMOOTHING)
    if isinstance(fg, (float, int)):
        fg = float(fg)
    if isinstance(ls, (float, int)):
        ls = float(ls)

    logging.info(
        "Чекпоинт: пауза %s, модальности %s, GNN %s",
        "вкл" if use_pause else "выкл",
        modalities,
        gnn_edge_mode,
    )

    data = dataset_utils.load_pickle(f"{pkl}")
    train_samples = data["train"]
    s0 = train_samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        pause_mu, pause_std = float(mu), float(std)
    else:
        pause_mu, pause_std = dataset_utils.compute_pause_norm_stats(train_samples)
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        logging.warning(
            "В pickle нет pause_norm_mu/std — mu/std по train. "
            "Обновите препроцесс."
        )
    dialogues_per_batch = ckpt.get(
        "dialogues_per_batch", arguments_and_constants.DIALOGUES_PER_BATCH
    )
    modality_feature_dim = ckpt.get("modality_feature_dim")
    if modality_feature_dim is None:
        modality_feature_dim = arguments_and_constants.DIMS[modalities]

    def _ds(samples):
        return Dataset(
            samples,
            dialogues_per_batch=dialogues_per_batch,
            modalities=modalities,
            modality_feature_dim=modality_feature_dim,
            pause_mu=pause_mu,
            pause_std=pause_std,
            augment=False,
            use_pause=use_pause,
        )

    test_dataset = _ds(data["test"])
    if args.tune_bias:
        dev_dataset = _ds(data["dev"])

    model = Apollo(
        modalities=modalities,
        device=arguments_and_constants.DEVICE,
        class_weights=cw,
        use_pause=use_pause,
        focal_gamma=fg,
        use_focal=use_f,
        label_smoothing=ls,
        use_heterogeneous_gnn=use_heterogeneous_gnn,
        **optional_apollo_kwargs_from_ma(ma),
    )
    model.to(arguments_and_constants.DEVICE)

    meta = _load_checkpoint_weights(
        model, ckpt_path, map_location=arguments_and_constants.DEVICE
    )
    if meta:
        logging.info("Checkpoint metadata: %s", {k: meta[k] for k in list(meta)[:5]})

    logit_bias: np.ndarray | None = None
    if args.tune_bias:
        logging.info(
            "Поиск logit-bias на dev (objective=%s, trials=%d) …",
            args.tune_bias_metric,
            args.bias_tune_trials,
        )
        g_dev, l_dev = collect_logits_and_golds(
            model, dev_dataset, arguments_and_constants.DEVICE, desc="dev logits"
        )
        n_cls = l_dev.shape[1]
        b, base_obj, tuned_obj = fit_logit_bias(
            l_dev,
            g_dev,
            n_classes=n_cls,
            n_trials=args.bias_tune_trials,
            seed=0,
            objective=args.tune_bias_metric,
        )
        naive_d = sklearn_metrics_numpy(
            g_dev, l_dev, dataset_constants.EMOTION_MAP, None, False
        )
        tune_d = sklearn_metrics_numpy(
            g_dev, l_dev, dataset_constants.EMOTION_MAP, b, False
        )
        logging.info(
            "dev: obj(%s) %.5f→%.5f | wF1 %.4f→%.4f macro %.4f→%.4f acc %.4f→%.4f | bias %s…",
            args.tune_bias_metric,
            base_obj,
            tuned_obj,
            naive_d["weighted_f1"],
            tune_d["weighted_f1"],
            naive_d["macro_f1"],
            tune_d["macro_f1"],
            naive_d["accuracy"],
            tune_d["accuracy"],
            np.round(b, 4)[: min(7, len(b))].tolist(),
        )
        logit_bias = b

    logging.info("Test:")
    pr = args.export_metrics_json is None
    test_metrics = evaluate_dataset(
        model,
        test_dataset,
        arguments_and_constants.DEVICE,
        dataset_constants.EMOTION_MAP,
        desc="test",
        print_report=pr,
        logit_bias=logit_bias,
    )
    if args.export_metrics_json is not None:
        out = {
            "checkpoint": str(ckpt_path.resolve()),
            "samples_pkl": str(pkl.resolve()),
            "modalities": modalities,
            "gnn_edge_mode": gnn_edge_mode,
            "use_pause": use_pause,
            "accuracy": float(test_metrics["accuracy"]),
            "weighted_f1": float(test_metrics["weighted_f1"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "mean_loss": float(test_metrics["mean_loss"]),
            "per_emotion_f1": {k: float(v) for k, v in test_metrics["per_class_f1"].items()},
        }
        args.export_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_metrics_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Метрики записаны в %s", args.export_metrics_json)


if __name__ == "__main__":
    main()
