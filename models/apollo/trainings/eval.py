"""
Evaluate a trained Apollo checkpoint on the test split.

Usage (from project root):
  PYTHONPATH=. python3 models/apollo/trainings/eval.py
  PYTHONPATH=. python3 models/apollo/trainings/eval.py --checkpoint results/apollo_meld_at_r01/model.pt
  # Подобрать сдвиг logit'ов на dev, затем отчёт на test (только dev участвует в подборе):
  PYTHONPATH=. python3 models/apollo/trainings/eval.py --tune-bias
"""
import argparse
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
    n_trials: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """
    Сдвиг logit'ов, подобранный на dev, чтобы argmax(softmax) менялся (T ≠ 1); полезен для
    смещения к распределению, близкому к test.
    """
    rng = np.random.default_rng(seed)
    y = gold.astype(np.int64)
    best_b = np.zeros(n_classes, dtype=np.float64)
    best = float((logits.argmax(1) == y).mean())
    for _ in range(n_trials):
        b = rng.uniform(-1.5, 1.5, size=n_classes)
        accv = float(((logits + b).argmax(1) == y).mean())
        if accv > best + 1e-9:
            best, best_b = accv, b.copy()
    return best_b


def _load_checkpoint_weights(model: Apollo, path: Path, map_location) -> dict:
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
    args = parser.parse_args()
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
        "Чекпоинт: пауза %s, модальности %s",
        "вкл" if use_pause else "выкл",
        modalities,
    )

    pkl = args.samples_pkl.resolve() if args.samples_pkl is not None else SAMPLES_PKL
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
    )
    model.to(arguments_and_constants.DEVICE)

    meta = _load_checkpoint_weights(
        model, ckpt_path, map_location=arguments_and_constants.DEVICE
    )
    if meta:
        logging.info("Checkpoint metadata: %s", {k: meta[k] for k in list(meta)[:5]})

    logit_bias: np.ndarray | None = None
    if args.tune_bias:
        logging.info("Поиск logit-bias на dev …")
        g_dev, l_dev = collect_logits_and_golds(
            model, dev_dataset, arguments_and_constants.DEVICE, desc="dev logits"
        )
        n_cls = l_dev.shape[1]
        b = fit_logit_bias(l_dev, g_dev, n_classes=n_cls, n_trials=25000, seed=0)
        d_acc = float((l_dev.argmax(1) == g_dev).mean())
        d_acc_b = float(((l_dev + b).argmax(1) == g_dev).mean())
        logging.info(
            "dev: accuracy %.4f → с bias %.4f | bias %s",
            d_acc,
            d_acc_b,
            np.round(b, 3).tolist(),
        )
        logit_bias = b

    logging.info("Test:")
    evaluate_dataset(
        model,
        test_dataset,
        arguments_and_constants.DEVICE,
        dataset_constants.EMOTION_MAP,
        desc="test",
        print_report=True,
        logit_bias=logit_bias,
    )


if __name__ == "__main__":
    main()
