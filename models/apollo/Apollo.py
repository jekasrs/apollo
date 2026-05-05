from __future__ import annotations

import torch
from torch import nn

from dataset.preprocess.utils import constants as dataset_constants
from models.apollo.utils.functions import batch_graphify
from models.apollo.utils.constants import RNN_HIDDEN_DIM, GNN_H1_DIM, GNN_H2_DIM, CLASSIFIER_HIDDEN_DIM, MODALITY_PROJ_DIM, \
    USE_INPUT_LAYER_NORM, DROPOUT_CLASSIFIER, USE_FOCAL_LOSS, FOCAL_GAMMA, LABEL_SMOOTHING, DROPOUT_RNN, \
    NUM_SEMANTIC_RELATIONS, DIMS
from models.apollo.inner_models.Classifier import Classifier
from models.apollo.inner_models.GNN import GNN
from models.apollo.inner_models.SeqContext import SeqContext


class Apollo(nn.Module):
    def __init__(
        self,
        modalities,
        device,
        class_weights=None,
        use_pause: bool = True,
        focal_gamma: float | None = None,
        use_focal: bool | None = None,
        label_smoothing: float | None = None,
    ):
        super(Apollo, self).__init__()
        self.modalities = modalities
        self.use_pause = use_pause
        u_dim = DIMS[modalities]
        g_dim = RNN_HIDDEN_DIM
        h1_dim = GNN_H1_DIM
        h2_dim = GNN_H2_DIM
        hc_dim = CLASSIFIER_HIDDEN_DIM
        self.device = device

        self.concat_gin_gout = True
        self.wp = 10
        self.wf = 10
        self.gnn_n_heads = 2

        if modalities == "at":
            self.audio_proj = nn.Linear(dataset_constants.AUDIO_FEATURE_DIM, MODALITY_PROJ_DIM)
            self.text_proj = nn.Linear(DIMS["t"], MODALITY_PROJ_DIM)
            rnn_content_dim = 2 * MODALITY_PROJ_DIM
        else:
            self.audio_proj = None
            self.text_proj = None
            rnn_content_dim = u_dim

        self.input_ln = nn.LayerNorm(rnn_content_dim) if USE_INPUT_LAYER_NORM else None
        rnn_in_dim = rnn_content_dim + (1 if use_pause else 0)

        self.rnn = SeqContext(
            dataset_embedding_dims=rnn_in_dim,
            hc_dim=g_dim,
            drop_rate=DROPOUT_RNN,
            seq_context_n_layer=2,
            device=device,
        )

        self.gnn = GNN(g_dim, h1_dim, h2_dim, NUM_SEMANTIC_RELATIONS, self.gnn_n_heads)
        if self.concat_gin_gout:
            classifier_input_dim = g_dim + h2_dim * self.gnn_n_heads
        else:
            classifier_input_dim = h2_dim * self.gnn_n_heads

        fg = FOCAL_GAMMA if focal_gamma is None else focal_gamma
        use_f = USE_FOCAL_LOSS if use_focal is None else use_focal
        ls = (
            0.0
            if use_f
            else (LABEL_SMOOTHING if label_smoothing is None else float(label_smoothing))
        )
        self.classifier = Classifier(
            classifier_input_dim,
            hc_dim,
            len(dataset_constants.EMOTION_MAP),
            drop_rate=DROPOUT_CLASSIFIER,
            class_weights=class_weights,
            use_focal=use_f,
            focal_gamma=fg,
            label_smoothing=ls,
        )

    def _prepare_input_tensor(self, x):
        if self.use_pause:
            pause = x[..., -1:]
            xc = x[..., :-1]
        else:
            pause = None
            xc = x
        if self.modalities == "at":
            audio = xc[..., :dataset_constants.AUDIO_FEATURE_DIM]
            text = xc[..., dataset_constants.AUDIO_FEATURE_DIM:]
            xc = torch.cat([self.audio_proj(audio), self.text_proj(text)], dim=-1)
        if self.input_ln is not None:
            xc = self.input_ln(xc)
        if self.use_pause:
            return torch.cat([xc, pause], dim=-1)
        return xc

    def _encode_dialogues(self, data):
        inp = self._prepare_input_tensor(data["input_tensor"])
        node_features = self.rnn(data["text_len_tensor"], inp)
        features, edge_index, edge_type = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.device,
        )
        graph_out = self.gnn(features, edge_index, edge_type)
        return features, graph_out

    def _per_utterance_features(self, features, graph_out, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        chunks = []
        start = 0
        for i in range(batch_size):
            length = text_len_tensor[i].item()
            token_feat = features[start : start + length]
            token_gnn = graph_out[start : start + length]
            if self.concat_gin_gout:
                combined = torch.cat([token_feat, token_gnn], dim=-1)
            else:
                combined = token_gnn
            chunks.append(combined)
            start += length
        return torch.cat(chunks, dim=0)

    def forward(self, data):
        features, graph_out = self._encode_dialogues(data)
        rep = self._per_utterance_features(
            features, graph_out, data["text_len_tensor"]
        )
        return self.classifier(rep)

    def get_loss(self, data):
        features, graph_out = self._encode_dialogues(data)
        rep = self._per_utterance_features(
            features, graph_out, data["text_len_tensor"]
        )
        return self.classifier.get_loss(rep, data["label_tensor"])
