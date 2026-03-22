import torch

LEARNING_RATE = 1e-4
MAX_GRAD_VALUE = 1.0
WEIGHT_DECAY = 1e-5
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32