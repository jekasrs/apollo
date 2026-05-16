import numpy as np
import torch


BATCH_KEY_UTTERANCE_TEXTS = "utterance_texts"
# Ключи батча, которые не переносятся на GPU
NON_TENSOR_BATCH_KEYS = frozenset({BATCH_KEY_UTTERANCE_TEXTS})


def batch_to_device(data: dict, device) -> None:
    for k, v in data.items():
        if k not in NON_TENSOR_BATCH_KEYS:
            data[k] = v.to(device)


def _append_similarity_edges(
    feat_block: torch.Tensor,
    cur_len: int,
    length_sum: int,
    edge_index: list,
    edge_type: list,
    *,
    topk: int,
    min_cos: float,
    similarity_type_id: int,
) -> None:
    """Рёбра по top-k косинусному сходству эмбеддингов реплик внутри одного диалога (оба направления)."""
    if topk <= 0 or cur_len < 2:
        return
    f = feat_block[:cur_len]
    fn = torch.nn.functional.normalize(f, p=2, dim=-1)
    sim = fn @ fn.t()
    sim.fill_diagonal_(-1.0)
    k = min(topk, cur_len - 1)
    vals, col_idx = sim.topk(k, dim=-1)
    for row in range(cur_len):
        for j in range(k):
            if vals[row, j] < min_cos:
                continue
            col = int(col_idx[row, j].item())
            if col == row:
                continue
            u = length_sum + row
            v = length_sum + col
            edge_index.append(torch.tensor([u, v], dtype=torch.long))
            edge_type.append(similarity_type_id)
            edge_index.append(torch.tensor([v, u], dtype=torch.long))
            edge_type.append(similarity_type_id)


def batch_graphify(
    features,
    lengths,
    speaker_tensor,
    wp,
    wf,
    device,
    *,
    similarity_topk: int = 0,
    similarity_min_cos: float = 0.35,
    num_relation_types: int = 4,
):
    """
    :param similarity_topk: число дополнительных соседей по cosine similarity на узел (0 = выкл.).
    :param num_relation_types: 4 — только временные типы; 5 — последний тип зарезервирован под similarity.
    """
    if similarity_topk > 0 and num_relation_types < 5:
        raise ValueError(
            "similarity_topk > 0 требует num_relation_types >= 5 (последний индекс — similarity)."
        )
    sim_type_id = num_relation_types - 1 if similarity_topk > 0 else -1

    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0

    for b in range(batch_size):
        cur_len = lengths[b].item()
        node_features.append(features[b, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]], dtype=torch.long))

            src, dst = item[0], item[1]
            sp1 = int(speaker_tensor[b, src].item())
            sp2 = int(speaker_tensor[b, dst].item())
            direction = 0 if src < dst else 1
            diff = 1 if sp1 != sp2 else 0
            etype = direction * 2 + diff
            edge_type.append(etype)

        if similarity_topk > 0:
            _append_similarity_edges(
                features[b],
                cur_len,
                length_sum,
                edge_index,
                edge_type,
                topk=similarity_topk,
                min_cos=similarity_min_cos,
                similarity_type_id=sim_type_id,
            )
        length_sum += cur_len

    node_features = torch.cat(node_features, dim=0).to(device)
    if not edge_index:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_type = torch.zeros((0,), dtype=torch.long, device=device)
    else:
        edge_index = torch.stack(edge_index).t().contiguous().to(device)
        edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)

    return node_features, edge_index, edge_type


def edge_perms(length, window_past, window_future):
    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[: min(length, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past) :]
        else:
            eff_array = array[
                max(0, j - window_past) : min(length, j + window_future + 1)
            ]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
