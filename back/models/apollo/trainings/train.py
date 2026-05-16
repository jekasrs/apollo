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
        help="Нет роста dev-метрики (--dev-metric) N эпох подряд — стоп. 0 = не использовать (все --epochs).",
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
        help="Label smoothing при --no-focal (по умолчанию 0.08, если не задано).",
    )
    parser.add_argument(
        "--dev-metric",
        type=str,
        choices=["weighted_f1", "accuracy", "macro_f1"],
        default="weighted_f1",
        help="По какой метрике dev сохраняется лучший чекпоинт и early stopping.",
    )
    parser.add_argument(
        "--audio-dev-weighted-f1",
        action="store_true",
        help=(
            "При --modalities a: выбирать лучший чекпоинт по dev weighted_f1. "
            "Иначе для «a» по умолчанию используется macro_f1 (weighted_f1 завышен из-за neutral)."
        ),
    )
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument(
        "--optimize-accuracy",
        action="store_true",
        help=(
            "Пресет: dev accuracy + weighted CE + label smoothing (--no-focal). "
            "Для большего акцента на accuracy: --uniform-class-weights."
        ),
    )
    preset.add_argument(
        "--optimize-weighted-f1",
        action="store_true",
        help=(
            "Пресет под рост weighted F1: dev-метрика weighted_f1, focal loss (как в constants), "
            "без label smoothing. Несовместим с --optimize-accuracy. Для редких классов полезно "
            "добавить --train-dup-fear-disgust 2."
        ),
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
        help="Повторить train-диалоги (см. --train-dup-emotions), где есть хотя бы одна из этих эмоций, N раз целиком. 0=выкл.",
    )
    parser.add_argument(
        "--train-dup-emotions",
        type=str,
        default="fear,disgust,sadness",
        help="Список имён классов через запятую (EMOTION_MAP), для --train-dup-fear-disgust > 0.",
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
        "--gnn-edge-mode",
        choices=("heterogeneous", "homogeneous"),
        default="heterogeneous",
        help="Первый слой GNN: RGCN по типам ребра vs GCN без типов (абляция «heterogeneous»).",
    )
    parser.add_argument(
        "--speaker-embed",
        action="store_true",
        help="Эмбеддинг локального id спикера в диалоге + проекция перед GNN (DialogueRNN/COGMEN-стиль).",
    )
    parser.add_argument("--speaker-emb-dim", type=int, default=32)
    parser.add_argument("--max-local-speakers", type=int, default=24)
    parser.add_argument(
        "--graph-similarity-topk",
        type=int,
        default=0,
        metavar="K",
        help="Добавить K рёбер на узел по cosine similarity контекстных эмбеддингов внутри диалога (0=выкл.; требует RGCN 5 типов).",
    )
    parser.add_argument(
        "--graph-similarity-min-cos",
        type=float,
        default=0.35,
        help="Мин. косинус для similarity-ребра.",
    )
    parser.add_argument(
        "--emotion-shift-weight",
        type=float,
        default=0.0,
        help="Вес вспомогательной CE-потери «смена эмоции» относительно предыдущей реплики (CFN-ESA-стиль). 0=выкл.",
    )
    parser.add_argument("--graph-wp", type=int, default=10, help="Окно графа: прошлые реплики.")
    parser.add_argument("--graph-wf", type=int, default=10, help="Окно графа: будущие реплики.")
    parser.add_argument(
        "--run-id",
        type=str,
        default="apollo_meld_at_r01",
        metavar="NAME",
        help="Папка results/<NAME>/: итоговый model.pt и лучшие веса по dev (см. results/README.md).",
    )
    args = parser.parse_args()

    if args.optimize_accuracy:
        prev_m = args.dev_metric
        args.dev_metric = "accuracy"
        if prev_m != "accuracy":
            logging.info(
                "optimize-accuracy: dev_metric принудительно accuracy (было %s)",
                prev_m,
            )
        args.no_focal = True
        logging.info(
            "optimize-accuracy: включены --dev-metric accuracy и --no-focal (CE + label smoothing)"
        )

    if args.optimize_weighted_f1:
        prev_m = args.dev_metric
        args.dev_metric = "weighted_f1"
        if prev_m != "weighted_f1":
            logging.info(
                "optimize-weighted-f1: dev_metric принудительно weighted_f1 (было %s)",
                prev_m,
            )
        if args.no_focal:
            logging.warning(
                "optimize-weighted-f1: отключаем --no-focal — для F1 оставляем focal loss"
            )
        args.no_focal = False
        logging.info(
            "optimize-weighted-f1: выбор чекпоинта по dev weighted_f1, focal loss ON"
        )

    if args.use_pause:
        use_pause = True
    elif args.no_pause:
        use_pause = False
    else:
        use_pause = arguments_and_constants.USE_PAUSE
    modalities = args.modalities or arguments_and_constants.MODALITIES

    coach_dev_metric = args.dev_metric
    if (
        modalities == "a"
        and not args.audio_dev_weighted_f1
        and args.dev_metric == "weighted_f1"
        and not args.optimize_accuracy
        and not args.optimize_weighted_f1
    ):
        coach_dev_metric = "macro_f1"
        logging.info(
            "Только аудио: лучший чекпоинт по dev macro_f1 (weighted_f1 часто выбирает модель «всё neutral»). "
            "Если нужен отбор по weighted_f1: флаг --audio-dev-weighted-f1."
        )

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
        else (2.5 if modalities == "a" else arguments_and_constants.FOCAL_GAMMA)
    )
    use_focal = not args.no_focal
    if args.label_smoothing is not None:
        label_smoothing = float(args.label_smoothing)
    else:
        label_smoothing = 0.08 if args.no_focal else arguments_and_constants.LABEL_SMOOTHING
    class_beta = (
        args.class_weight_beta
        if args.class_weight_beta is not None
        else (0.996 if modalities == "a" else arguments_and_constants.CLASS_WEIGHT_BETA)
    )

    if modalities == "a" and args.focal_gamma is None:
        logging.info(
            "Только аудио: focal_gamma=%s (дефолт выше базового — против коллапса в majority-класс)",
            focal_gamma,
        )
    if modalities == "a" and args.class_weight_beta is None:
        logging.info(
            "Только аудио: class_weight_beta=%s (сильнее редкие классы, чем базовый %.4f)",
            class_beta,
            arguments_and_constants.CLASS_WEIGHT_BETA,
        )

    use_heterogeneous_gnn = args.gnn_edge_mode == "heterogeneous"
    logging.info("GNN edge mode: %s", args.gnn_edge_mode)
    if args.graph_similarity_topk > 0 and not use_heterogeneous_gnn:
        raise SystemExit(
            "--graph-similarity-topk > 0 несовместимо с --gnn-edge-mode homogeneous "
            "(нужны типизированные рёбра RGCN для отдельного типа similarity)."
        )

    apollo_extras = dict(
        use_speaker_embedding=bool(args.speaker_embed),
        speaker_emb_dim=int(args.speaker_emb_dim),
        max_local_speakers=int(args.max_local_speakers),
        graph_similarity_topk=int(args.graph_similarity_topk),
        graph_similarity_min_cos=float(args.graph_similarity_min_cos),
        emotion_shift_loss_weight=float(args.emotion_shift_weight),
        graph_wp=int(args.graph_wp),
        graph_wf=int(args.graph_wf),
    )
    logging.info("Apollo graph extras: %s", apollo_extras)

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
        names = [x.strip().lower() for x in args.train_dup_emotions.split(",") if x.strip()]
        rare: set[int] = set()
        for n in names:
            if n not in dataset_constants.EMOTION_MAP:
                raise SystemExit(
                    f"--train-dup-emotions: неизвестная эмоция {n!r}. Допустимо: {sorted(dataset_constants.EMOTION_MAP)}"
                )
            rare.add(dataset_constants.EMOTION_MAP[n])
        train_samples, n_add_utts, n_affected = duplicate_train_dialogues_by_labels(
            train_samples, rare, args.train_dup_fear_disgust
        )
        logging.info(
            "Дублирование диалогов с эмоциями %s: +%d реплик, затронуто %d диалогов (копий на диалог: %d)",
            names,
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
        use_heterogeneous_gnn=use_heterogeneous_gnn,
        **apollo_extras,
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
        dev_select_metric=coach_dev_metric,
        checkpoint_save_dir=run_dir,
    )

    logging.log(logging.INFO, "Apollo model created successfully")

    logging.info("Starting training")
    best_dev_score, best_epoch, best_state, train_losses, dev_f1s, test_f1s = coach.train()
    w_f1_at_best = dev_f1s[best_epoch - 1] if best_epoch and 1 <= best_epoch <= len(dev_f1s) else None

    checkpoint = {
        "best_dev_score": best_dev_score,
        "dev_select_metric": coach_dev_metric,
        "best_dev_f1": w_f1_at_best
        if w_f1_at_best is not None
        else (float(best_dev_score) if coach_dev_metric == "weighted_f1" else None),
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
            "gnn_edge_mode": args.gnn_edge_mode,
            "optimize_accuracy": bool(args.optimize_accuracy),
            "optimize_weighted_f1": bool(args.optimize_weighted_f1),
            "single_modality_audio_projection": modalities == "a",
            **apollo_extras,
        },
        "dialogues_per_batch": arguments_and_constants.DIALOGUES_PER_BATCH,
        "modality_feature_dim": arguments_and_constants.DIMS[modalities],
        "use_pause": use_pause,
        "train_dup_fear_disgust": int(args.train_dup_fear_disgust),
        "train_dup_emotions": str(args.train_dup_emotions),
        "results_run_dir": str(run_dir),
        "run_id": args.run_id,
    }

    out = run_dir / "model.pt"
    torch.save(checkpoint, out)
    logging.info("Чекпоинт сохранён: %s", out)


if __name__ == "__main__":
    main()
