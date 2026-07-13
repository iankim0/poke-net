def build_tiny_subset(paths: list[str], species: list[str], images_per_species: int = 10) -> list[int]:
    """Deterministic ~5-species x ~10-real-image subset. Picks the first `images_per_species`
    real_* images (sorted filename order) for each species in `species`. Returns global indices
    into `paths` (and therefore into the matching features tensor)."""
    species_set = set(species)
    per_species_indices: dict[str, list[int]] = {s: [] for s in species}

    for idx, p in enumerate(paths):
        s, filename = p.split("/", 1)
        if s in species_set and filename.startswith("real_"):
            per_species_indices[s].append(idx)  # paths is sorted, so this is too

    subset_indices = []
    for s in species:
        subset_indices.extend(per_species_indices[s][:images_per_species])
    return subset_indices
