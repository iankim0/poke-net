import torch
import torch.nn.functional as F


def cosine_topk_accuracy(features: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Precision@k: for each row, cosine-NN the k closest *other* rows and compute the fraction
    that share its label, averaged over all rows. features: (N, D) raw (not required to be
    pre-normalized). labels: (N,)."""
    normalized = F.normalize(features, p=2, dim=1)
    similarity = normalized @ normalized.t()  # (N, N)
    similarity.fill_diagonal_(float("-inf"))  # exclude self as its own neighbor

    topk_indices = similarity.topk(k, dim=1).indices  # (N, k)
    topk_labels = labels[topk_indices]  # (N, k)

    matches = (topk_labels == labels.unsqueeze(1)).float()  # (N, k)
    per_query_precision = matches.mean(dim=1)  # (N,)
    return per_query_precision.mean().item()
