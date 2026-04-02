"""
Evaluate a trained Apollo checkpoint on dev or test split.

Usage (from project root):
  python eval.py
"""
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from Dataset import SAMPLES_PATH
from Dataset.models.Apollo import Apollo
from Dataset.models.Dataset import Dataset
from Dataset.models.constants import DEVICE, DIALOGUES_PER_BATCH, MODALITIES
from Dataset.utils.constants import DIMS, EMOTION_MAP
from Dataset.models.functions import batch_to_device
from Dataset.utils.io_utils import load_pickle
from Dataset.utils.pause_stats import compute_pause_norm_stats

logging.basicConfig(level=logging.INFO)


def evaluate_dataset(model, dataset, device, label_to_idx, desc="eval", print_report=False):
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
    pred_labels = np.argmax(logits, axis=1)
    weighted_f1 = metrics.f1_score(golds, pred_labels, average="weighted", zero_division=0)
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
        print(
            metrics.classification_report(
                golds,
                pred_labels,
                target_names=list(label_to_idx.keys()),
                digits=4,
                zero_division=0,
            )
        )
        print(f"Accuracy: {acc:.4f}  Weighted F1: {weighted_f1:.4f}  Loss (sum/n_batches): {mean_loss:.4f}")

    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "mean_loss": mean_loss,
        "per_class_f1": per_class_f1,
    }


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
    ckpt = torch.load(Path("checkpoints/model.pt"), map_location=DEVICE, weights_only=False)
    cw = ckpt.get("class_weights")
    if cw is not None:
        cw = cw.to(DEVICE)

    data = load_pickle(Path("Dataset") / SAMPLES_PATH)
    samples = data["test"]

    s0 = samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        pause_mu, pause_std = float(mu), float(std)
    else:
        pause_mu, pause_std = compute_pause_norm_stats(data["train"])
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        logging.warning(
            "В pickle нет pause_norm_mu/std — mu/std по train. "
            "Обновите данные: python Dataset/preprocess.py"
        )
    dialogues_per_batch = ckpt.get("dialogues_per_batch", DIALOGUES_PER_BATCH)
    modality_feature_dim = ckpt.get("modality_feature_dim", DIMS[MODALITIES])

    dataset = Dataset(
        samples,
        dialogues_per_batch=dialogues_per_batch,
        modalities=MODALITIES,
        modality_feature_dim=modality_feature_dim,
        pause_mu=pause_mu,
        pause_std=pause_std,
        augment=False,
    )

    model = Apollo(modalities=MODALITIES, device=DEVICE, class_weights=cw)
    model.to(DEVICE)

    meta = _load_checkpoint_weights(model, Path("checkpoints/model.pt"), map_location=DEVICE)
    if meta:
        logging.info("Checkpoint metadata: %s", {k: meta[k] for k in list(meta)[:5]})

    evaluate_dataset(
        model,
        dataset,
        DEVICE,
        EMOTION_MAP,
        desc="test",
        print_report=True,
    )


if __name__ == "__main__":
    main()
