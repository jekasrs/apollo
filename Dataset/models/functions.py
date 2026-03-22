import numpy as np
import torch

def batch_graphify(
    features,
    lengths,
    speaker_tensor,
    wp,
    wf,
    edge_type_to_idx,
    device,
    n_speaker_buckets=None,
):
    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []

    def spk_id(x):
        if n_speaker_buckets is None:
            return int(x)
        return int(x) % n_speaker_buckets

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))

            speaker1 = spk_id(speaker_tensor[j, item[0]].item())
            speaker2 = spk_id(speaker_tensor[j, item[1]].item())
            direction = 0 if item[0] < item[1] else 1
            edge_type.append(edge_type_to_idx[(speaker1, speaker2, direction)])

    node_features = torch.cat(node_features, dim=0).to(device)
    edge_index = torch.stack(edge_index).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type).long().to(device)
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)

    return node_features, edge_index, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[: min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past) :]
        else:
            eff_array = array[
                max(0, j - window_past) : min(length, j + window_future + 1)
            ]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
