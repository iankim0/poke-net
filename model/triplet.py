import torch
import torch.nn.functional as F


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Squared-Euclidean pairwise distances. (B, 128) -> (B, B). Diagonal zeroed, clamped >= 0."""
    dot_product = embeddings @ embeddings.t()  # (B, B), dot_product[i, j] = e_i . e_j
    sq_norms = torch.diagonal(dot_product)  # (B,), sq_norms[i] = e_i . e_i = ||e_i||^2

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
    distances = sq_norms.unsqueeze(0) + sq_norms.unsqueeze(1) - 2.0 * dot_product

    distances = distances.clamp(min=0.0)  # fp error can make same/near-same vectors slightly negative
    distances.fill_diagonal_(0.0)  # d(i, i) must be exactly 0
    return distances


def embedding_spread(embeddings: torch.Tensor) -> float:
    """Mean pairwise distance across *all* pairs in the batch (not just mined pos/neg). Embeddings
    are L2-normalized (unit hypersphere), so this trending toward 0 means the batch is collapsing
    toward a single direction regardless of species label — the collapse failure mode."""
    distances = pairwise_distances(embeddings)
    n = embeddings.size(0)
    return (distances.sum() / (n * (n - 1))).item()


def anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, B) bool. [i, j] True iff j is a valid positive for anchor i: same label, i != j."""
    same_label = labels.unsqueeze(1) == labels.unsqueeze(0)  # [i, j] = labels[i] == labels[j]
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return same_label & not_self


def anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, B) bool. [i, j] True iff j is a valid negative for anchor i: different label."""
    return labels.unsqueeze(1) != labels.unsqueeze(0)


def _batch_hard(distances: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, margin: float):
    # push invalid entries to the end of the range they can't win: -inf for positives (we take a max),
    # +inf for negatives (we take a min). Never zero-fill; 0 looks like a real (small) distance.
    hardest_pos_dist = distances.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    hardest_neg_dist = distances.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    triplet_loss = (hardest_pos_dist - hardest_neg_dist + margin).clamp(min=0.0)

    stats = {
        "frac_active": (triplet_loss > 1e-12).float().mean().item(),
        "mean_pos_dist": hardest_pos_dist.mean().item(),
        "mean_neg_dist": hardest_neg_dist.mean().item(),
    }
    return triplet_loss.mean(), stats


def _batch_soft(distances: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
    # same hardest-pos/hardest-neg mining as _batch_hard; only the loss shape changes.
    hardest_pos_dist = distances.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    hardest_neg_dist = distances.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    # softplus(x) = log(1 + exp(x)) — smooth hinge, no clamp, no explicit margin.
    # softplus(0) = log(2) =~ 0.693 is this loss's collapse-plateau analog to hard-margin's `margin`.
    triplet_loss = F.softplus(hardest_pos_dist - hardest_neg_dist)

    stats = {
        "frac_active": (triplet_loss > 1e-12).float().mean().item(),
        "mean_pos_dist": hardest_pos_dist.mean().item(),
        "mean_neg_dist": hardest_neg_dist.mean().item(),
    }
    return triplet_loss.mean(), stats


def _mine_semihard(distances: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
    # same hardest-positive mining as _batch_hard; negative mining changes.
    hardest_pos_dist = distances.masked_fill(~pos_mask, float("-inf")).max(dim=1).values  # (B,)

    # semi-hard negatives: farther than the anchor's hardest positive (already-wrong, but not extreme).
    semihard_mask = neg_mask & (distances > hardest_pos_dist.unsqueeze(1))
    has_semihard = semihard_mask.any(dim=1)

    semihard_neg_dist = distances.masked_fill(~semihard_mask, float("inf")).min(dim=1).values
    # fallback for anchors with no semi-hard candidate in this batch: plain hardest negative.
    hardest_neg_dist = distances.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    neg_dist = torch.where(has_semihard, semihard_neg_dist, hardest_neg_dist)

    return hardest_pos_dist, neg_dist, has_semihard


def _batch_semihard(distances: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, margin: float):
    hardest_pos_dist, neg_dist, has_semihard = _mine_semihard(distances, pos_mask, neg_mask)

    triplet_loss = (hardest_pos_dist - neg_dist + margin).clamp(min=0.0)

    stats = {
        "frac_active": (triplet_loss > 1e-12).float().mean().item(),
        "mean_pos_dist": hardest_pos_dist.mean().item(),
        "mean_neg_dist": neg_dist.mean().item(),
        "frac_semihard_available": has_semihard.float().mean().item(),
    }
    return triplet_loss.mean(), stats


def _batch_semihard_soft(distances: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
    hardest_pos_dist, neg_dist, has_semihard = _mine_semihard(distances, pos_mask, neg_mask)

    # softplus(x) = log(1 + exp(x)) — same smooth hinge as _batch_soft, on semi-hard-mined negatives.
    triplet_loss = F.softplus(hardest_pos_dist - neg_dist)

    stats = {
        "frac_active": (triplet_loss > 1e-12).float().mean().item(),
        "mean_pos_dist": hardest_pos_dist.mean().item(),
        "mean_neg_dist": neg_dist.mean().item(),
        "frac_semihard_available": has_semihard.float().mean().item(),
    }
    return triplet_loss.mean(), stats


def batch_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2, strategy: str = "hard"):
    """(B,128), (B,) -> (scalar loss, stats dict). Assumes every anchor has >=1 valid positive and
    >=1 valid negative in the batch (guaranteed by a well-formed PK batch, P>=2 species, K>=2 images)."""
    distances = pairwise_distances(embeddings)
    pos_mask = anchor_positive_mask(labels)
    neg_mask = anchor_negative_mask(labels)

    if strategy == "hard":
        return _batch_hard(distances, pos_mask, neg_mask, margin)
    if strategy == "soft":
        return _batch_soft(distances, pos_mask, neg_mask)
    if strategy == "semihard":
        return _batch_semihard(distances, pos_mask, neg_mask, margin)
    if strategy == "semihard_soft":
        return _batch_semihard_soft(distances, pos_mask, neg_mask)
    raise NotImplementedError(f"strategy={strategy!r} not implemented yet")
