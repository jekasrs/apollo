import torch
from torch import nn
from Dataset.models.Classifier import Classifier
from Dataset.models.GNN import GNN
from Dataset.models.SeqContext import SeqContext
from Dataset.models.functions import batch_graphify
from Dataset.models.constants import (
    CLASSIFIER_HIDDEN_DIM,
    FOCAL_GAMMA,
    GNN_SPEAKER_BUCKETS,
    LABEL_SMOOTHING,
    MODALITY_PROJ_DIM,
    USE_FOCAL_LOSS,
    USE_INPUT_LAYERNORM,
)
from Dataset.utils.constants import AUDIO_FEATURE_DIM, DIMS, EMOTION_MAP, SPEAKER_MAP


class Apollo(nn.Module):
    def __init__(self, modalities, device, class_weights=None):
        super(Apollo, self).__init__()
        self.modalities = modalities
        u_dim = DIMS[modalities]   # Размерность входных features utterances (до проекций)
        g_dim = 200   # Размерность выхода RNN
        h1_dim = 150  # Размерность скрытого слоя GNN
        h2_dim = 150  # Размерность выхода GNN
        hc_dim = CLASSIFIER_HIDDEN_DIM
        self.dataset_label_dict = EMOTION_MAP
        self.dataset_speaker_dict = SPEAKER_MAP
        self.device = device

        # Конфигурационные параметры
        self.concat_gin_gout = True  # Конкатенировать вход и выход GNN
        self.wp = 10  # Окно прошлого (window past)
        self.wf = 10  # Окно будущего (window future)
        self.gnn_n_heads = 2  # Количество голов attention в GNN
        # RGCN relation count is 2 * n^2; optional bucketing (see constants.SMOKE_TEST).
        self.n_speakers_gnn = (
            GNN_SPEAKER_BUCKETS if GNN_SPEAKER_BUCKETS is not None else len(self.dataset_speaker_dict)
        )
        self._bucket_speakers = GNN_SPEAKER_BUCKETS is not None

        if modalities == "at":
            self.audio_proj = nn.Linear(AUDIO_FEATURE_DIM, MODALITY_PROJ_DIM)
            self.text_proj = nn.Linear(DIMS["t"], MODALITY_PROJ_DIM)
            rnn_in_dim = 2 * MODALITY_PROJ_DIM
        else:
            self.audio_proj = None
            self.text_proj = None
            rnn_in_dim = u_dim

        self.input_ln = nn.LayerNorm(rnn_in_dim) if USE_INPUT_LAYERNORM else None

        # SeqContext контекстуализация реплик(utterances), чтобы каждая реплика понимала, что было до и после неё.
        self.rnn = SeqContext(
            dataset_embedding_dims=rnn_in_dim,
            hc_dim=g_dim,
            drop_rate=0.3,
            seq_context_n_layer=2,
            device=device,
        )

        # GNN распознает связи между репликами (+1 измерение 'сложности' модели)
        self.gnn = GNN(g_dim, h1_dim, h2_dim, self.n_speakers_gnn, self.gnn_n_heads)
        if self.concat_gin_gout:
            classifier_input_dim = g_dim + h2_dim * self.gnn_n_heads
        else:
            classifier_input_dim = h2_dim * self.gnn_n_heads

        # Classifier определяет к какому классу относится реплика
        self.classifier = Classifier(
            classifier_input_dim,
            hc_dim,
            len(EMOTION_MAP),
            drop_rate=0.3,
            class_weights=class_weights,
            use_focal=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_GAMMA,
            label_smoothing=0.0 if USE_FOCAL_LOSS else LABEL_SMOOTHING,
        )
        # Словарь для типов ребер графа
        self.edge_type_to_idx = self._create_edge_type_mapping(self.n_speakers_gnn)

    def _create_edge_type_mapping(self, n_speakers):
        """Создает mapping типов ребер для графа (tuple: speaker1, speaker2, direction)"""
        edge_type_to_idx = {}
        idx = 0
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_to_idx[(j, k, 0)] = idx  # forward
                idx += 1
                edge_type_to_idx[(j, k, 1)] = idx  # backward
                idx += 1
        return edge_type_to_idx

    def _prepare_input_tensor(self, x):
        if self.modalities == "at":
            audio = x[..., :AUDIO_FEATURE_DIM]
            text = x[..., AUDIO_FEATURE_DIM:]
            x = torch.cat([self.audio_proj(audio), self.text_proj(text)], dim=-1)
        if self.input_ln is not None:
            x = self.input_ln(x)
        return x

    def get_rep(self, data):
        """Получение представлений utterances"""
        inp = self._prepare_input_tensor(data["input_tensor"])
        # Контекстуализация через RNN
        node_features = self.rnn(data["text_len_tensor"], inp)
        # Построение графа диалога
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
            n_speaker_buckets=self.n_speakers_gnn if self._bucket_speakers else None,
        )
        # Обработка графа через GNN
        graph_out = self.gnn(features, edge_index, edge_type)
        return node_features, features, graph_out, edge_index_lengths

    def forward(self, data):
        node_features, features, graph_out, edge_index_lengths = self.get_rep(data)

        # Усреднение по токенам внутри каждой реплики
        batch_size = data["text_len_tensor"].size(0)
        rep_features = []
        start = 0
        for i in range(batch_size):
            length = data["text_len_tensor"][i].item()
            # усредняем только те токены, которые есть
            token_feat = features[start:start + length]
            token_gnn = graph_out[start:start + length]
            if self.concat_gin_gout:
                combined = torch.cat([token_feat, token_gnn], dim=-1)
            else:
                combined = token_gnn
            rep_features.append(combined.mean(dim=0))  # усредняем по токенам
            start += length

        rep_features = torch.stack(rep_features)

        # классификация
        out = self.classifier(rep_features)
        return out

    def get_loss(self, data):
        node_features, features, graph_out, edge_index_lengths = self.get_rep(data)

        batch_size = data["text_len_tensor"].size(0)
        rep_features = []
        start = 0
        for i in range(batch_size):
            length = data["text_len_tensor"][i].item()
            token_feat = features[start:start + length]
            token_gnn = graph_out[start:start + length]
            if self.concat_gin_gout:
                combined = torch.cat([token_feat, token_gnn], dim=-1)
            else:
                combined = token_gnn
            rep_features.append(combined.mean(dim=0))
            start += length

        rep_features = torch.stack(rep_features)
        loss = self.classifier.get_loss(rep_features, data["label_tensor"])
        return loss