import argparse
import logging
from pathlib import Path

import torch

from dataset import SAMPLES_PKL
from dataset.models.Dataset import Dataset
from dataset.oversample_dialogues import duplicate_train_dialogues_by_labels
from dataset.preprocess.utils import utils as dataset_utils
from dataset.preprocess.utils import constants as dataset_constants
from models.apollo.Apollo import Apollo
from models.apollo.Coach import Coach
from models.apollo.Optim import Optim
from models.apollo.utils import constants as arguments_and_constants
from models.apollo.utils.class_weights import compute_class_weights_from_samples
from models.apollo.utils.repo_paths import repo_root

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Обучение Apollo: с каналом паузы или без (см. USE_PAUSE в constants)."
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--use-pause",
        action="store_true",
        help="Включить нормализованную паузу до следующей реплики (переопределяет USE_PAUSE)",
    )
    g.add_argument(
        "--no-pause",
        action="store_true",
        help="Обучать без канала паузы (переопределяет USE_PAUSE)",
    )
    parser.add_argument(
        "--modalities",
        choices=["a", "t", "at"],
        default=None,
        help="Модальности: a / t / at (по умолчанию — из constants)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Макс. число эпох (по умолчанию — EPOCHS в constants).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Нет роста dev weighted F1 N эпох — стоп. 0 = не использовать (все --epochs).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="AdamW lr (по умолчанию LEARNING_RATE в constants).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Гамма focal loss (по умолчанию FOCAL_GAMMA в constants).",
    )
    parser.add_argument(
        "--no-focal",
        action="store_true",
        help="Вместо focal — weighted cross-entropy с label_smoothing (часто выше accuracy).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Label smoothing при --no-focal (по умолчанию 0.05).",
    )
    parser.add_argument(
        "--dev-metric",
        type=str,
        choices=["weighted_f1", "accuracy", "macro_f1"],
        default="weighted_f1",
        help="По какой метрике dev сохраняется лучший чекпоинт и early stopping.",
    )
    parser.add_argument(
        "--uniform-class-weights",
        action="store_true",
        help="Равные веса классов в loss (максимизировать raw accuracy, без rebalancing).",
    )
    parser.add_argument(
        "--class-weight-beta",
        type=float,
        default=None,
        help="beta для effective-number весов (по умолчанию CLASS_WEIGHT_BETA; выше=мягче).",
    )
    parser.add_argument(
        "--train-dup-fear-disgust",
        type=int,
        default=0,
        metavar="N",
        help="Повторить train-диалоги, где есть fear или disgust, N раз целиком (реальные MELD-реплики, новый dialogue_id). 0=выкл.",
    )
    parser.add_argument(
        "--samples-pkl",
        type=Path,
        default=None,
        help="Путь к samples.pkl (по умолчанию SAMPLES_PKL; см. env APOLLO_SAMPLES_PKL).",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Инициализировать веса из сохранённого train-чекпоинта (ключ best_state), напр. MELD перед IEMOCAP.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="apollo_meld_at_r01",
        metavar="NAME",
        help="Папка results/<NAME>/: итоговый model.pt и лучшие веса по dev (см. results/README.md).",
    )
    args = parser.parse_args()
    if args.use_pause:
        use_pause = True
    elif args.no_pause:
        use_pause = False
    else:
        use_pause = arguments_and_constants.USE_PAUSE
    modalities = args.modalities or arguments_and_constants.MODALITIES
    epochs = args.epochs if args.epochs is not None else arguments_and_constants.EPOCHS
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else arguments_and_constants.EARLY_STOPPING_PATIENCE
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else arguments_and_constants.LEARNING_RATE
    )
    focal_gamma = (
        args.focal_gamma
        if args.focal_gamma is not None
        else arguments_and_constants.FOCAL_GAMMA
    )
    use_focal = not args.no_focal
    if args.label_smoothing is not None:
        label_smoothing = float(args.label_smoothing)
    else:
        label_smoothing = 0.08 if args.no_focal else arguments_and_constants.LABEL_SMOOTHING
    class_beta = (
        args.class_weight_beta
        if args.class_weight_beta is not None
        else arguments_and_constants.CLASS_WEIGHT_BETA
    )

    run_dir = (repo_root() / "results" / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Каталог прогона: %s", run_dir)

    pkl = args.samples_pkl.resolve() if args.samples_pkl is not None else SAMPLES_PKL
    data = dataset_utils.load_pickle(f"{pkl}")
    logging.info("Loaded data: %s", pkl)

    train_samples: list = list(data["train"])
    dev_samples = data["dev"]
    test_samples = data["test"]

    if args.train_dup_fear_disgust > 0:
        rare = {
            dataset_constants.EMOTION_MAP["fear"],
            dataset_constants.EMOTION_MAP["disgust"],
        }
        train_samples, n_add_utts, n_affected = duplicate_train_dialogues_by_labels(
            train_samples, rare, args.train_dup_fear_disgust
        )
        logging.info(
            "Дублирование диалогов с fear/disgust: +%d реплик, затронуто %d диалогов (копий на диалог: %d)",
            n_add_utts,
            n_affected,
            args.train_dup_fear_disgust,
        )

    s0 = train_samples[0]
    mu = getattr(s0, "pause_norm_mu", None)
    std = getattr(s0, "pause_norm_std", None)
    if mu is not None and std is not None:
        pause_mu, pause_std = float(mu), float(std)
    else:
        pause_mu, pause_std = dataset_utils.compute_pause_norm_stats(train_samples)
        pause_mu, pause_std = float(pause_mu), float(pause_std)
        logging.warning(
            "В pickle нет pause_norm_mu/std — пересчитано по train. "
            "Обновите данные: python Dataset/preprocess.py"
        )
    ds_common = dict(
        dialogues_per_batch=arguments_and_constants.DIALOGUES_PER_BATCH,
        modalities=modalities,
        modality_feature_dim=arguments_and_constants.DIMS[modalities],
        pause_mu=pause_mu,
        pause_std=pause_std,
        use_pause=use_pause,
    )
    train_set = Dataset(train_samples, augment=True, **ds_common)
    dev_set = Dataset(dev_samples, augment=False, **ds_common)
    test_set = Dataset(test_samples, augment=False, **ds_common)

    logging.info("Режим паузы: %s", "включён" if use_pause else "выключен")
    logging.info("Модальности: %s", modalities)
    logging.info(
        "План: до %d эпох, early stopping patience=%d (0=выкл.)",
        epochs,
        early_stopping_patience,
    )
    logging.info(
        "learning_rate=%s focal_gamma=%s use_focal=%s label_smoothing=%s dev_metric=%s",
        learning_rate,
        focal_gamma,
        use_focal,
        label_smoothing,
        args.dev_metric,
    )
    logging.log(logging.INFO, f"Train batches: {len(train_set)}")
    logging.log(logging.INFO, f"Dev batches: {len(dev_set)}")
    logging.log(logging.INFO, f"Test batches: {len(test_set)}")

    if args.uniform_class_weights:
        class_weights = None
        logging.info("Class weights: отключены (равный вклад классов).")
    else:
        class_weights = compute_class_weights_from_samples(
            train_samples,
            len(dataset_constants.EMOTION_MAP),
            arguments_and_constants.DEVICE,
            beta=class_beta,
        )
        logging.info(
            "Class weights (balanced, beta=%s): %s",
            class_beta,
            dict(zip(dataset_constants.EMOTION_MAP.keys(), class_weights.tolist())),
        )

    logging.log(logging.INFO, "Started creating Apollo model")
    model = Apollo(
        modalities=modalities,
        device=arguments_and_constants.DEVICE,
        class_weights=class_weights,
        use_pause=use_pause,
        focal_gamma=focal_gamma,
        use_focal=use_focal,
        label_smoothing=label_smoothing,
    )
    model.to(arguments_and_constants.DEVICE)
    if args.from_checkpoint is not None:
        ck = torch.load(
            str(args.from_checkpoint),
            map_location=arguments_and_constants.DEVICE,
            weights_only=False,
        )
        st = ck.get("best_state")
        if st is None:
            raise SystemExit("В чекпоинте нет best_state; нужен файл, сохранённый train.py (results/<run>/model.pt)")
        model.load_state_dict(st, strict=True)
        logging.info("Веса загружены из %s", args.from_checkpoint)

    opt = Optim(learning_rate, arguments_and_constants.MAX_GRAD_VALUE, arguments_and_constants.WEIGHT_DECAY)
    opt.set_parameters(params=model.parameters(), name="adamw")
    scheduler = opt.get_scheduler("reduceLR")

    coach = Coach(
        train=train_set,
        dev=dev_set,
        test=test_set,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        epochs=epochs,
        device=arguments_and_constants.DEVICE,
        label_to_idx=dataset_constants.EMOTION_MAP,
        run_test_each_epoch=arguments_and_constants.RUN_TEST_EACH_EPOCH,
        early_stopping_patience=early_stopping_patience,
        dev_select_metric=args.dev_metric,
        checkpoint_save_dir=run_dir,
    )

    logging.log(logging.INFO, "Apollo model created successfully")

    logging.info("Starting training")
    best_dev_score, best_epoch, best_state, train_losses, dev_f1s, test_f1s = coach.train()
    w_f1_at_best = dev_f1s[best_epoch - 1] if best_epoch and 1 <= best_epoch <= len(dev_f1s) else None

    checkpoint = {
        "best_dev_score": best_dev_score,
        "dev_select_metric": args.dev_metric,
        "best_dev_f1": w_f1_at_best
        if w_f1_at_best is not None
        else (float(best_dev_score) if args.dev_metric == "weighted_f1" else None),
        "best_epoch": best_epoch,
        "learning_rate": learning_rate,
        "best_state": best_state,
        "train_losses": train_losses,
        "dev_f1s": dev_f1s,
        "test_f1s": test_f1s,
        "class_weights": class_weights.detach().cpu() if class_weights is not None else None,
        "class_weight_beta": class_beta,
        "uniform_class_weights": args.uniform_class_weights,
        "model_args": {
            "modalities": modalities,
            "modality_proj_dim": arguments_and_constants.MODALITY_PROJ_DIM,
            "classifier_hidden_dim": arguments_and_constants.CLASSIFIER_HIDDEN_DIM,
            "use_input_layernorm": arguments_and_constants.USE_INPUT_LAYER_NORM,
            "use_focal_loss": use_focal,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
            "use_pause": use_pause,
        },
        "dialogues_per_batch": arguments_and_constants.DIALOGUES_PER_BATCH,
        "modality_feature_dim": arguments_and_constants.DIMS[modalities],
        "use_pause": use_pause,
        "train_dup_fear_disgust": int(args.train_dup_fear_disgust),
        "results_run_dir": str(run_dir),
        "run_id": args.run_id,
    }

    out = run_dir / "model.pt"
    torch.save(checkpoint, out)
    logging.info("Чекпоинт сохранён: %s", out)


if __name__ == "__main__":
    main()
