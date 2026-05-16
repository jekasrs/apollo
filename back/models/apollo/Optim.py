"""
Обертка над оптиизаторо py torch
Создает оптимизатор
Управляет learning rate
Обрезает градиенты
Подключает scheduler
"""
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR


class Optim:
    def __init__(self, lr, max_grad_value, weight_decay):
        self.lr = lr # насколько сильно енять веса
        self.max_grad_value = max_grad_value # gradient clipping
        self.weight_decay = weight_decay # штрафы за большие веса
        self.params = None
        self.optimizer = None

    def set_parameters(self, params, name):
        """
        :param params: Параметры оптимизатора
        :param name: Название оптиизатора (sgd, rmsprop, adam, adamw)
        """
        self.params = list(params)
        if name == "sgd": # классический градиентный спуск
            self.optimizer = optim.SGD(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "rmsprop": # адаптивный learning rate
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adam":
            self.optimizer = optim.Adam(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adamw": # улучшенная версия Adam
            self.optimizer = optim.AdamW(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )

    def get_scheduler(self, sch):
        """
        Scheduler меняте learning rate во время обучения
        если loss не уменьшается, уменьшаем lr
        :param sch: название стратегии
        """
        sched = None
        if sch == "reduceLR":
            sched = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=1e-7,
            )
        elif sch == "expLR":
            sched = ExponentialLR(self.optimizer, gamma=0.9)
        return sched

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        if self.max_grad_value != -1:
            clip_grad_value_(self.params, self.max_grad_value) # защита от взрыва градиента
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
