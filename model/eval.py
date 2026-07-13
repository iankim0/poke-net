from collections import Counter

import torch


def species_scores(sims: torch.Tensor, ref_labels: torch.Tensor, num_species: int) -> torch.Tensor:
    """sims: (n_query, n_ref) cosine similarities against reference vectors. ref_labels: (n_ref,)
    species index per reference. Dedupes multi-reference matches to one score per species (best/max
    similarity across that species' references). Returns (n_query, num_species) of per-species scores."""
    n_query = sims.size(0)
    species_best = torch.full((n_query, num_species), float("-inf"), dtype=sims.dtype, device=sims.device)
    index = ref_labels.unsqueeze(0).expand(n_query, -1)
    species_best.scatter_reduce_(1, index, sims, reduce="amax")
    return species_best


def species_ranking(sims: torch.Tensor, ref_labels: torch.Tensor, num_species: int) -> torch.Tensor:
    """Ranks species per query by deduped max similarity (see species_scores), best first.
    Returns (n_query, num_species) of species indices."""
    return species_scores(sims, ref_labels, num_species).argsort(dim=1, descending=True)


def topk_accuracy(ranked_species: torch.Tensor, query_labels: torch.Tensor, k: int) -> float:
    """ranked_species: (n_query, num_species) species indices, best-first (from species_ranking).
    query_labels: (n_query,) true species index per query. Fraction of queries whose true species
    appears anywhere in the top k."""
    topk = ranked_species[:, :k]
    correct = (topk == query_labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def confused_pairs(ranked_species: torch.Tensor, query_labels: torch.Tensor, species_names: list[str]) -> list[tuple[str, str, int]]:
    """Top-1 misclassifications only. Returns (true_species, predicted_species, count) sorted by
    count descending, one entry per distinct wrong (true, predicted) pair seen among the queries."""
    top1 = ranked_species[:, 0]
    wrong = top1 != query_labels

    pairs = Counter(
        (species_names[true.item()], species_names[pred.item()])
        for true, pred in zip(query_labels[wrong], top1[wrong])
    )
    return [(true, pred, count) for (true, pred), count in pairs.most_common()]
