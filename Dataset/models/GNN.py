"""Графовая нейронная сеть, которая обрабатывает граф:
Граф состоит из:
    1. nodes(узлы) - основная информация о реплике
    2. edges(ребра) - связь между узлами
    3. типы связей (relations) - кто кому говорит
        - Например, ответ влияет иначе чем вопрос
"""
from torch import nn
from torch_geometric.nn import RGCNConv, TransformerConv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, n_speakers, gnn_n_heads):
        """
        Инициализация
        :param g_dim: Размер входных признаков узла (длина вектора)
        :param h1_dim: Размер после первого слоя
        :param h2_dim: Размер после второго слоя
        :param n_speakers: кол-во участников в диалоге
        :param gnn_n_heads: кол-во attention голов механизмов (кол-во способов оценить соседей)
        """
        super(GNN, self).__init__()
        self.num_relations = 2 * n_speakers ** 2 # все возможные пары в обоих направлениях
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=gnn_n_heads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * gnn_n_heads)

    def forward(self, node_features, edge_index, edge_type):
        """
        Forward проход для обучения модели
        :param node_features: Признаки узлов
        :param edge_index: Список ребер
        :param edge_type: Тип для каждого ребра
        """
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = nn.functional.leaky_relu(x)
        return x