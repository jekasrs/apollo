"""Графовая сеть: узлы — реплики; типы рёбер — семантические (время × совпадение спикера)."""
from torch import nn
from torch_geometric.nn import RGCNConv, TransformerConv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, gnn_n_heads):
        """
        :param num_relations: число типов для RGCN (фиксировано, без глобального словаря спикеров).
        """
        super(GNN, self).__init__()
        self.num_relations = num_relations
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=gnn_n_heads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * gnn_n_heads)

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = nn.functional.leaky_relu(x)
        return x
