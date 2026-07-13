import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Yields batches of P*K training indices: P species, K images per species. Pair with
    DataLoader(dataset, batch_sampler=pk_sampler). `labels` should already be restricted to the
    training split — this sampler has no notion of train/val, it just groups by label."""

    def __init__(self, labels: torch.Tensor, P: int, K: int, num_batches: int):
        self.P = P
        self.K = K
        self.num_batches = num_batches

        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            self.label_to_indices[label].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

        if P > len(self.unique_labels):
            raise ValueError(f"P={P} exceeds available species ({len(self.unique_labels)})")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_species = random.sample(self.unique_labels, self.P)
            batch = []
            for species in batch_species:
                indices = self.label_to_indices[species]
                if len(indices) >= self.K:
                    batch.extend(random.sample(indices, self.K))
                else:
                    batch.extend(random.choices(indices, k=self.K))  # fewer than K: sample with replacement
            yield batch

    def __len__(self):
        return self.num_batches
