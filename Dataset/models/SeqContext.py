from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqContext(nn.Module):
    """
    Рекуррентная сеть учитывает порядок и помнит контекст из LSTM.
    Каждая реплика должна понимать, что было до и после неё.
    """
    def __init__(self, dataset_embedding_dims, hc_dim, drop_rate, seq_context_n_layer, device):
        """

        :param dataset_embedding_dims: размер входных данных (вектор 768 BERT)
        :param hc_dim: Размер скрытого состояния слоя
        :param drop_rate:
        :param seq_context_n_layer: кол-во слоев LSTM
        :param device:
        """
        super(SeqContext, self).__init__()
        self.input_size = dataset_embedding_dims
        self.hidden_dim = hc_dim
        self.dropout = nn.Dropout(drop_rate)
        self.device = device

        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_dim // 2, # одна слева направо, другая справа налево: bidirectional=True
            dropout=drop_rate,
            bidirectional=True,
            num_layers=seq_context_n_layer,
            batch_first=True,
        )

    def forward(self, text_len_tensor, text_tensor):
        """Forward проход для обучения модели"""
        packed = pack_padded_sequence(
            text_tensor, text_len_tensor, batch_first=True, enforce_sorted=False
        )
        rnn_out, (_, _) = self.rnn(packed, None)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out
