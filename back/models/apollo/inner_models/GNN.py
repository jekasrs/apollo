"""Графовая сеть: узлы — реплики; типы рёбер — семантические (время × совпадение спикера)."""
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv, TransformerConv


class GNN(nn.Module):
    def __init__(
        self,
        g_dim,
        h1_dim,
        h2_dim,
        num_relations,
        gnn_n_heads,
        use_relation_types: bool = True,
    ):
        """
        :param num_relations: число типов для RGCN (если ``use_relation_types``).
        :param use_relation_types: True — разные веса по типам ребра (RGCN, «heterogeneous»).
                                  False — GCN первого слоя, тип ребра не используется.
        """
        super(GNN, self).__init__()
        self.num_relations = num_relations
        self.use_relation_types = use_relation_types
        if use_relation_types:
            self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)
        else:
            self.conv1 = GCNConv(g_dim, h1_dim)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=gnn_n_heads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * gnn_n_heads)

    def forward(self, node_features, edge_index, edge_type):
        if self.use_relation_types:
            x = self.conv1(node_features, edge_index, edge_type)
        else:
            x = self.conv1(node_features, edge_index)
        x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = nn.functional.leaky_relu(x)
        return x
