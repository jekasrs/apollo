import torch

from dataset.preprocess.utils.constants import AUDIO_FEATURE_DIM, TEXT_EMBED_DIM

MODALITIES = "a"
# Канал нормализованной паузы (log1p + z-score по train) в конце входа BiLSTM
USE_PAUSE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIALOGUES_PER_BATCH = 20
EPOCHS = 50
# 0 = без early stopping (все EPOCHS). Иначе остановка, если dev weighted F1 не растёт N эпох подряд.
EARLY_STOPPING_PATIENCE = 5
RUN_TEST_EACH_EPOCH = True

LEARNING_RATE = 2e-4
MAX_GRAD_VALUE = 1.0
WEIGHT_DECAY = 1e-4
CLASS_WEIGHT_BETA = 0.999

# Улучшения для F1 (мультимодальность и дисбаланс классов)
MODALITY_PROJ_DIM = 320
CLASSIFIER_HIDDEN_DIM = 256
RNN_HIDDEN_DIM = 320
GNN_H1_DIM = 192
GNN_H2_DIM = 192
DROPOUT_RNN = 0.2
DROPOUT_CLASSIFIER = 0.2
USE_INPUT_LAYER_NORM = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 1.5
LABEL_SMOOTHING = 0.0

# Аугментация (только train, см. Coach)
USE_TRAIN_AUGMENTATION = True
AUG_APPLY_PROB = 0.65
AUG_AUDIO_STD = 0.028
AUG_TEXT_STD = 0.028



# RGCN: (время вперёд / назад) × (тот же спикер / другой)
NUM_SEMANTIC_RELATIONS = 4

# a=768, t=768, at=1536 размер входа для каждой модальности
DIMS = {"a": AUDIO_FEATURE_DIM, "t": TEXT_EMBED_DIM, "at": TEXT_EMBED_DIM + AUDIO_FEATURE_DIM}