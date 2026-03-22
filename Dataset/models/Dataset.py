import math
import random
import torch


class Dataset:
    def __init__(self, samples, batch_size, modalities, dataset_embedding_dims) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.modalities = modalities
        self.embedding_dim = dataset_embedding_dims

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()

        input_tensor = torch.zeros((batch_size, mx, self.embedding_dim))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            utterances.append(s.sentence)
            tmp = []
            for t, a in zip(s.sbert_sentence_embeddings, s.audio):
                t = torch.tensor(t)
                a = torch.tensor(a)
                if self.modalities == "at":
                    tmp.append(torch.cat((a, t)))
                elif self.modalities == "a":
                    tmp.append(a)
                elif self.modalities == "t":
                    tmp.append(t)

            tmp = torch.stack(tmp)
            input_tensor[i, :cur_len, :] = tmp
            speaker_tensor[i, :cur_len] = torch.tensor([s.speaker])

            labels.extend(s.label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "input_tensor": input_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
        }
        return data

    def shuffle(self):
        random.shuffle(self.samples)
