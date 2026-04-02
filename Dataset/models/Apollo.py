import torch
from torch import nn
from Dataset.models.Classifier import Classifier
from Dataset.models.GNN import GNN
from Dataset.models.SeqContext import SeqContext
from Dataset.models.functions import NUM_SEMANTIC_RELATIONS, batch_graphify
from Dataset.models.constants import (
    CLASSIFIER_HIDDEN_DIM,
    DROPOUT_CLASSIFIER,
    DROPOUT_RNN,
    FOCAL_GAMMA,
    GNN_H1_DIM,
    GNN_H2_DIM,
    LABEL_SMOOTHING,
    MODALITY_PROJ_DIM,
    RNN_HIDDEN_DIM,
    USE_FOCAL_LOSS,
    USE_INPUT_LAYER_NORM,
)
from Dataset.utils.constants import AUDIO_FEATURE_DIM, DIMS, EMOTION_MAP


class Apollo(nn.Module):
    def __init__(self, modalities, device, class_weights=None):
        super(Apollo, self).__init__()
        self.modalities = modalities
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
            self.audio_proj = nn.Linear(AUDIO_FEATURE_DIM, MODALITY_PROJ_DIM)
            self.text_proj = nn.Linear(DIMS["t"], MODALITY_PROJ_DIM)
            rnn_content_dim = 2 * MODALITY_PROJ_DIM
        else:
            self.audio_proj = None
            self.text_proj = None
            rnn_content_dim = u_dim

        self.input_ln = nn.LayerNorm(rnn_content_dim) if USE_INPUT_LAYER_NORM else None
        rnn_in_dim = rnn_content_dim + 1

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

        self.classifier = Classifier(
            classifier_input_dim,
            hc_dim,
            len(EMOTION_MAP),
            drop_rate=DROPOUT_CLASSIFIER,
            class_weights=class_weights,
            use_focal=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_GAMMA,
            label_smoothing=0.0 if USE_FOCAL_LOSS else LABEL_SMOOTHING,
        )

    def _prepare_input_tensor(self, x):
        pause = x[..., -1:]
        xc = x[..., :-1]
        if self.modalities == "at":
            audio = xc[..., :AUDIO_FEATURE_DIM]
            text = xc[..., AUDIO_FEATURE_DIM:]
            xc = torch.cat([self.audio_proj(audio), self.text_proj(text)], dim=-1)
        if self.input_ln is not None:
            xc = self.input_ln(xc)
        return torch.cat([xc, pause], dim=-1)

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
