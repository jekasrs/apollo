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
    BATCH_SIZE = 8
    EPOCHS = 50
    TRAIN_MAX_SAMPLES = None
    DEV_MAX_SAMPLES = None
    TEST_MAX_SAMPLES = None
    RUN_TEST_EACH_EPOCH = True
    GNN_SPEAKER_BUCKETS = None


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')