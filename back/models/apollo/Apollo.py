from __future__ import annotations

import torch
import torch.nn.functional as F
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
        use_heterogeneous_gnn: bool = True,
        *,
        use_speaker_embedding: bool = False,
        speaker_emb_dim: int = 32,
        max_local_speakers: int = 24,
        graph_similarity_topk: int = 0,
        graph_similarity_min_cos: float = 0.35,
        emotion_shift_loss_weight: float = 0.0,
        graph_wp: int = 10,
        graph_wf: int = 10,
    ):
        super(Apollo, self).__init__()
        self.modalities = modalities
        self.use_pause = use_pause
        self.use_heterogeneous_gnn = use_heterogeneous_gnn
        self.use_speaker_embedding = use_speaker_embedding
        self.max_local_speakers = max(2, int(max_local_speakers))
        self.graph_similarity_topk = int(graph_similarity_topk)
        self.graph_similarity_min_cos = float(graph_similarity_min_cos)
        self.emotion_shift_loss_weight = float(emotion_shift_loss_weight)
        self.wp = int(graph_wp)
        self.wf = int(graph_wf)

        u_dim = DIMS[modalities]
        g_dim = RNN_HIDDEN_DIM
        h1_dim = GNN_H1_DIM
        h2_dim = GNN_H2_DIM
        hc_dim = CLASSIFIER_HIDDEN_DIM
        self.device = device

        self.concat_gin_gout = True
        self.gnn_n_heads = 2

        self._gnn_num_relations = NUM_SEMANTIC_RELATIONS + (
            1 if self.graph_similarity_topk > 0 else 0
        )

        if modalities == "at":
            self.audio_proj = nn.Linear(dataset_constants.AUDIO_FEATURE_DIM, MODALITY_PROJ_DIM)
            self.text_proj = nn.Linear(DIMS["t"], MODALITY_PROJ_DIM)
            rnn_content_dim = 2 * MODALITY_PROJ_DIM
        elif modalities == "a":
            # Как в ветке at: сжимаем L2-эмбеддинги Wav2Vec до MODALITY_PROJ_DIM — иначе сеть часто
            # схлопывается в предсказание majority-класса (neutral) при слабом аудиосигнале.
            self.audio_proj = nn.Linear(dataset_constants.AUDIO_FEATURE_DIM, MODALITY_PROJ_DIM)
            self.text_proj = None
            rnn_content_dim = MODALITY_PROJ_DIM
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

        spk_dim = int(speaker_emb_dim)
        if self.use_speaker_embedding:
            self.speaker_emb = nn.Embedding(self.max_local_speakers, spk_dim)
            self.pre_gnn_proj = nn.Linear(g_dim + spk_dim, g_dim)
        else:
            self.speaker_emb = None
            self.pre_gnn_proj = None

        self.gnn = GNN(
            g_dim,
            h1_dim,
            h2_dim,
            self._gnn_num_relations,
            self.gnn_n_heads,
            use_relation_types=use_heterogeneous_gnn,
        )
        if self.concat_gin_gout:
            classifier_input_dim = g_dim + h2_dim * self.gnn_n_heads
        else:
            classifier_input_dim = h2_dim * self.gnn_n_heads

        self.shift_head = (
            nn.Linear(classifier_input_dim, 2)
            if self.emotion_shift_loss_weight > 0
            else None
        )

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
        elif self.modalities == "a":
            audio = xc[..., :dataset_constants.AUDIO_FEATURE_DIM]
            xc = self.audio_proj(audio)
        if self.input_ln is not None:
            xc = self.input_ln(xc)
        if self.use_pause:
            return torch.cat([xc, pause], dim=-1)
        return xc

    def _encode_dialogues(self, data):
        inp = self._prepare_input_tensor(data["input_tensor"])
        rnn_out = self.rnn(data["text_len_tensor"], inp)
        if self.use_speaker_embedding:
            spk_ids = data["speaker_tensor"].clamp(0, self.max_local_speakers - 1)
            spk_e = self.speaker_emb(spk_ids)
            features = self.pre_gnn_proj(torch.cat([rnn_out, spk_e], dim=-1))
        else:
            features = rnn_out

        features, edge_index, edge_type = batch_graphify(
            features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.device,
            similarity_topk=self.graph_similarity_topk,
            similarity_min_cos=self.graph_similarity_min_cos,
            num_relation_types=self._gnn_num_relations,
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

    def _emotion_shift_targets(
        self, label_tensor: torch.Tensor, text_len_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Переключение эмоции относительно предыдущей реплики в том же диалоге (-100 = маска)."""
        tgt = torch.full_like(label_tensor, -100)
        offset = 0
        for i in range(text_len_tensor.size(0)):
            L = int(text_len_tensor[i].item())
            labs = label_tensor[offset : offset + L]
            for t in range(1, L):
                tgt[offset + t] = (labs[t] != labs[t - 1]).long()
            offset += L
        return tgt

    def get_loss(self, data):
        features, graph_out = self._encode_dialogues(data)
        rep = self._per_utterance_features(
            features, graph_out, data["text_len_tensor"]
        )
        main = self.classifier.get_loss(rep, data["label_tensor"])
        if self.shift_head is None:
            return main
        shift_logits = self.shift_head(rep)
        shift_tgt = self._emotion_shift_targets(
            data["label_tensor"], data["text_len_tensor"]
        )
        shift_loss = F.cross_entropy(shift_logits, shift_tgt, ignore_index=-100)
        return main + self.emotion_shift_loss_weight * shift_loss
