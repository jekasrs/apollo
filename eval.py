"""
Evaluate a trained Apollo checkpoint on dev or test split.

Usage (from project root):
  python eval.py
  python eval.py --checkpoint checkpoints/model.pt --split test
  python eval.py --max-samples 256
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
from Dataset.models.constants import DEVICE, BATCH_SIZE, TEST_MAX_SAMPLES, MODALITIES
from Dataset.utils.constants import DIMS, EMOTION_MAP
from Dataset.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO)


def evaluate_dataset(model, dataset, device, label_to_idx, desc="eval", print_report=False):
    """
    Run the model on all batches of a Dataset and return metrics.
    Used by training and by eval.py.
    """
    model.eval()
    total_loss = 0.0
    golds = []
    preds = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc):
            data = dataset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if k != "utterance_texts":
                    data[k] = v.to(device)
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
    """Load weights from train.py checkpoint or legacy saves. Returns extra metadata dict."""
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

    if TEST_MAX_SAMPLES is not None:
        samples = samples[: TEST_MAX_SAMPLES]

    dataset = Dataset(
        samples,
        batch_size=BATCH_SIZE,
        modalities=MODALITIES,
        dataset_embedding_dims=DIMS[MODALITIES],
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
