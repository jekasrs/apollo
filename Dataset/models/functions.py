import numpy as np
import torch

# RGCN: (время вперёд / назад) × (тот же спикер / другой)
NUM_SEMANTIC_RELATIONS = 4

BATCH_KEY_UTTERANCE_TEXTS = "utterance_texts"
# Ключи батча, которые не переносятся на GPU
NON_TENSOR_BATCH_KEYS = frozenset({BATCH_KEY_UTTERANCE_TEXTS})


def batch_to_device(data: dict, device) -> None:
    for k, v in data.items():
        if k not in NON_TENSOR_BATCH_KEYS:
            data[k] = v.to(device)


def batch_graphify(
    features,
    lengths,
    speaker_tensor,
    wp,
    wf,
    device,
):
    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0

    for b in range(batch_size):
        cur_len = lengths[b].item()
        node_features.append(features[b, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))

            src, dst = item[0], item[1]
            sp1 = int(speaker_tensor[b, src].item())
            sp2 = int(speaker_tensor[b, dst].item())
            direction = 0 if src < dst else 1
            diff = 1 if sp1 != sp2 else 0
            etype = direction * 2 + diff
            edge_type.append(etype)

    node_features = torch.cat(node_features, dim=0).to(device)
    edge_index = torch.stack(edge_index).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type).long().to(device)

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
