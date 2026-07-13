from collections import defaultdict

import torch


def build_species_labels(paths: list[str]) -> tuple[torch.Tensor, list[str]]:
    """paths like '001_bulbasaur/real_0001.jpg' -> (label per path, sorted species names, index = label)."""
    species_names = sorted({p.split("/", 1)[0] for p in paths})
    species_to_idx = {name: i for i, name in enumerate(species_names)}
    labels = torch.tensor([species_to_idx[p.split("/", 1)[0]] for p in paths], dtype=torch.long)
    return labels, species_names


def build_train_mask(paths: list[str], val_per_species: int = 10) -> torch.Tensor:
    """True = train. Holds out the first `val_per_species` real_* images per species (by sorted
    filename order) for val; aug_* images are always train."""
    species_to_real_indices: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(paths):
        species, filename = p.split("/", 1)
        if filename.startswith("real_"):
            species_to_real_indices[species].append(idx)  # paths is already sorted, so this is too

    val_indices = [idx for indices in species_to_real_indices.values() for idx in indices[:val_per_species]]

    is_train = torch.ones(len(paths), dtype=torch.bool)
    is_train[val_indices] = False
    return is_train
