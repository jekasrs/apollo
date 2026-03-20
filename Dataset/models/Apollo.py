import torch
from torch import nn
from Dataset.models.Classifier import Classifier
from Dataset.models.GNN import GNN
from Dataset.models.SeqContext import SeqContext
from Dataset.models.functions import batch_graphify
from Dataset.utils.constants import EMOTION_MAP


class Apollo(nn.Module):
    def __init__(self, device):
        super(Apollo, self).__init__()
        u_dim = 100   # Размерность входных features utterances
        g_dim = 200   # Размерность выхода RNN
        h1_dim = 150  # Размерность скрытого слоя GNN
        h2_dim = 150  # Размерность выхода GNN
        hc_dim = 100  # Размерность скрытого слоя классификатора
        self.dataset_label_dict = EMOTION_MAP
        self.device = device

        # Конфигурационные параметры
        self.concat_gin_gout = True  # Конкатенировать вход и выход GNN
        self.wp = 10  # Окно прошлого (window past)
        self.wf = 10  # Окно будущего (window future)
        self.gnn_n_heads = 2  # Количество голов attention в GNN
        self.n_speakers = 2  # Количество уникальных speakers

        # SeqContext контекстуализация реплик(utterances), чтобы каждая реплика понимала, что было до и после неё.
        self.rnn = SeqContext(
            dataset_embedding_dims=u_dim,
            hc_dim=g_dim,
            drop_rate=0.3,
            seq_context_n_layer=2,
            device = device,
        )

        # GNN распознает связи между репликами (+1 измерение 'сложности' модели)
        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.n_speakers, self.gnn_n_heads)
        if self.concat_gin_gout:
            classifier_input_dim = g_dim + h2_dim * self.gnn_n_heads
        else:
            classifier_input_dim = h2_dim * self.gnn_n_heads

        # Classifier определяет к какому классу относится реплика
        self.classifier = Classifier(classifier_input_dim, hc_dim, len(EMOTION_MAP), drop_rate=0.3, class_weights=None)

        # Словарь для типов ребер графа
        self.edge_type_to_idx = self._create_edge_type_mapping(self.n_speakers)

    def _create_edge_type_mapping(self, n_speakers):
        """Создает mapping типов ребер для графа"""
        edge_type_to_idx = {}
        idx = 0
        for j in range(n_speakers):
            for k in range(n_speakers):
                # Два типа ребер для каждой пары speakers (0/1 для разных направлений)
                edge_type_to_idx[f"{j}{k}0"] = idx
                idx += 1
                edge_type_to_idx[f"{j}{k}1"] = idx
                idx += 1
        return edge_type_to_idx

    def get_rep(self, data):
        """Получение представлений utterances"""
        # Контекстуализация через RNN/Transformer
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])
        # Построение графа диалога
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device
        )
        # Обработка графа через GNN
        graph_out = self.gnn(features, edge_index, edge_type)
        return graph_out, features

    def forward(self, data):
        """Прямой проход - предсказание"""
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            # Конкатенация исходных и преобразованных features
            combined = torch.cat([features, graph_out], dim=-1)
            out = self.classifier(combined)
        else:
            out = self.classifier(graph_out)
        return out

    def get_loss(self, data):
        """Вычисление loss"""
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            combined = torch.cat([features, graph_out], dim=-1)
            loss = self.classifier.get_loss(combined, data["label_tensor"])
        else:
            loss = self.classifier.get_loss(graph_out, data["label_tensor"])
        return loss