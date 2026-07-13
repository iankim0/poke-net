import torch
import torch.nn.functional as F

from model.feature_cache import load_cache
from model.dataset import build_species_labels, build_train_mask
from model.pk_sampler import PKSampler
from model.head import ProjectionHead
from model.triplet import batch_triplet_loss

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
P, K = 16, 4
MARGIN = 0.2

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this check"
    device = "cuda"

    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]

    labels, species_names = build_species_labels(paths)
    is_train = build_train_mask(paths, val_per_species=2)

    train_indices = is_train.nonzero(as_tuple=True)[0]
    train_labels = labels[train_indices]

    sampler = PKSampler(train_labels, P=P, K=K, num_batches=1)
    batch_local = next(iter(sampler))  # indices into train_indices/train_labels
    global_indices = train_indices[torch.tensor(batch_local)]

    batch_features = features[global_indices].to(device)
    batch_labels = train_labels[torch.tensor(batch_local)].to(device)

    head = ProjectionHead(feature_dim=cache["feature_dim"], embedding_dim=128).to(device)
    embeddings = F.normalize(head(batch_features), p=2, dim=1)

    loss, stats = batch_triplet_loss(embeddings, batch_labels, margin=MARGIN, strategy="hard")
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in head.parameters() if p.grad is not None]
    n_head_params = sum(1 for _ in head.parameters())

    print("batch size:", len(batch_local))
    print("loss:", loss.item())
    print("stats:", stats)
    print("loss is finite and > 0:", torch.isfinite(loss).item() and loss.item() > 0)
    print("embeddings contain NaN:", torch.isnan(embeddings).any().item())
    print(f"head params with gradient: {len(grad_norms)}/{n_head_params}")
    print("head grad norms:", grad_norms)
    print("all head grads nonzero:", all(g > 0 for g in grad_norms))
