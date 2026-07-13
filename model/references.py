import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def build_references(embeddings: torch.Tensor, labels: torch.Tensor, num_species: int, k: int = 5):
    """Cluster each species' embeddings into up to k groups (k-means) and return each cluster's
    re-normalized centroid as a reference vector. embeddings: (N, D) L2-normalized, e.g. train-only.
    labels: (N,) species indices in [0, num_species). Returns (ref_embeddings (R, D), ref_labels (R,))
    with R = sum over species of min(k, images available for that species) — species with fewer than
    k embeddings get one reference per embedding rather than erroring."""
    ref_embeddings = []
    ref_labels = []

    for species in range(num_species):
        species_embeddings = embeddings[labels == species]
        n = species_embeddings.size(0)
        effective_k = min(k, n)

        if effective_k == 1:
            centroids = species_embeddings.mean(dim=0, keepdim=True)
        else:
            kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=0)
            kmeans.fit(species_embeddings.cpu().numpy())
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=embeddings.dtype, device=embeddings.device)

        ref_embeddings.append(F.normalize(centroids, p=2, dim=1))
        ref_labels.extend([species] * effective_k)

    ref_embeddings = torch.cat(ref_embeddings, dim=0)
    return ref_embeddings, torch.tensor(ref_labels, dtype=torch.long, device=ref_embeddings.device)
