import torch

LEARNING_RATE = 1e-4
MAX_GRAD_VALUE = 1.0
WEIGHT_DECAY = 1e-5

# Set False for full-quality training (slower). True = fast sanity check.
SMOKE_TEST = True

MODALITIES = "t"

if SMOKE_TEST:
    BATCH_SIZE = 32
    EPOCHS = 10
    TRAIN_MAX_SAMPLES = None
    DEV_MAX_SAMPLES = None
    TEST_MAX_SAMPLES = None
    RUN_TEST_EACH_EPOCH = False
    GNN_SPEAKER_BUCKETS = 16
else:
    BATCH_SIZE = 16
    EPOCHS = 50
    TRAIN_MAX_SAMPLES = None
    DEV_MAX_SAMPLES = None
    TEST_MAX_SAMPLES = None
    RUN_TEST_EACH_EPOCH = True
    GNN_SPEAKER_BUCKETS = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Улучшения для F1 (мультимодальность и дисбаланс классов) ---
# Для режима "at": отдельные линейные проекции текста и аудио в общее пространство (вместо сырой конкатенации в BiLSTM).
MODALITY_PROJ_DIM = 256
# Скрытый слой классификатора (было 100).
CLASSIFIER_HIDDEN_DIM = 128
# LayerNorm по признакам перед BiLSTM (стабилизирует масштаб модальностей).
USE_INPUT_LAYERNORM = True
# Focal loss для редких эмоций; при True label_smoothing не используется.
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
# Если USE_FOCAL_LOSS = False, можно включить сглаживание меток (0.05).
LABEL_SMOOTHING = 0.0
