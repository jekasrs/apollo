"""Классификатор MLP(Multi-Layer Perceptron)"""
from torch import nn
import torch.nn.functional as F
import torch


class Classifier(nn.Module):

    def __init__(self, input_dim, hidden_size, num_classes, drop_rate, class_weights):
        """

        :param input_dim: Размер входного вектора
        :param hidden_size: Размер скрытого слоя
        :param num_classes: Кол-во классов
        :param drop_rate: Параметр для борьбы с переобучением
        :param class_weights: веса классов (редки классы должны получать больший штраф за ошибку)
        """
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.class_weights = class_weights

        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(drop_rate)
        self.lin2 = nn.Linear(hidden_size, num_classes)

        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, h):
        hidden = F.relu(self.lin1(h))
        hidden = self.drop(hidden)
        scores = self.lin2(hidden)
        return scores

    def get_prob(self, h):
        scores = self.forward(h)
        log_probs = F.log_softmax(scores, dim=1)
        return log_probs

    def get_loss(self, h, label_tensor):
        scores = self.forward(h)
        loss = self.ce_loss(scores, label_tensor)
        return loss

    def predict(self, h):
        scores = self.forward(h)
        probs = F.softmax(scores, dim=1)
        y_hat = torch.argmax(probs, dim=1)
        return y_hat
