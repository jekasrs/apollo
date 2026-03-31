"""Классификатор MLP(Multi-Layer Perceptron)"""
from torch import nn
import torch.nn.functional as F
import torch


class Classifier(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_size,
        num_classes,
        drop_rate,
        class_weights,
        use_focal=False,
        focal_gamma=2.0,
        label_smoothing=0.0,
    ):
        """
        :param input_dim: Размер входного вектора
        :param hidden_size: Размер скрытого слоя
        :param num_classes: Кол-во классов
        :param drop_rate: Параметр для борьбы с переобучением
        :param class_weights: веса классов (редки классы должны получать больший штраф за ошибку)
        :param use_focal: focal loss вместо CE (лучше на редких классах)
        :param focal_gamma: гамма focal loss
        :param label_smoothing: сглаживание меток (используется только при use_focal=False)
        """
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = float(label_smoothing)

        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(drop_rate)
        self.lin2 = nn.Linear(hidden_size, num_classes)

        if class_weights is not None:
            self.register_buffer("ce_class_weights", class_weights.float())
        else:
            self.ce_class_weights = None

    def forward(self, h):
        hidden = F.relu(self.lin1(h))
        hidden = self.drop(hidden)
        scores = self.lin2(hidden)
        return scores

    def get_prob(self, h):
        scores = self.forward(h)
        log_probs = F.log_softmax(scores, dim=1)
        return log_probs

    def _focal_loss(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8)
        loss = -((1.0 - pt) ** self.focal_gamma) * log_pt
        w = self.ce_class_weights
        if w is not None:
            loss = loss * w[target]
        return loss.mean()

    def get_loss(self, h, label_tensor):
        scores = self.forward(h)
        if self.use_focal:
            return self._focal_loss(scores, label_tensor)
        w = self.ce_class_weights
        ls = self.label_smoothing if self.label_smoothing > 0 else 0.0
        return F.cross_entropy(scores, label_tensor, weight=w, label_smoothing=ls)

    def predict(self, h):
        scores = self.forward(h)
        probs = F.softmax(scores, dim=1)
        y_hat = torch.argmax(probs, dim=1)
        return y_hat
