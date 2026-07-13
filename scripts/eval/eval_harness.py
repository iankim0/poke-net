import sys
import time

import torch
import torch.nn.functional as F

from model.dataset import build_species_labels, build_train_mask
from model.eval import confused_pairs, species_ranking, topk_accuracy
from model.feature_cache import load_cache
from model.head import ProjectionHead
from model.references import build_references
from model.train import load_checkpoint

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
DEFAULT_CHECKPOINT_PATH = "checkpoints/full_semihard_head.pt"
NUM_REFS_PER_SPECIES = 5
KS = (1, 5, 10)


def run_eval(name: str, train_embeddings: torch.Tensor, train_labels: torch.Tensor,
             val_embeddings: torch.Tensor, val_labels: torch.Tensor, num_species: int,
             species_names: list[str]) -> dict:
    start = time.time()
    ref_emb, ref_labels = build_references(train_embeddings, train_labels, num_species=num_species, k=NUM_REFS_PER_SPECIES)
    sims = val_embeddings @ ref_emb.t()
    ranked = species_ranking(sims, ref_labels, num_species=num_species)

    results = {k: topk_accuracy(ranked, val_labels, k=k) for k in KS}
    elapsed = time.time() - start

    print(f"\n{name} (train-ref k={NUM_REFS_PER_SPECIES} / val-query, {elapsed:.1f}s):")
    for k in KS:
        print(f"  top-{k}: {results[k]:.4f}")

    pairs = confused_pairs(ranked, val_labels, species_names)
    print(f"  top-1 misclassifications: {len(pairs)} distinct pairs, {sum(c for _, _, c in pairs)} total errors")
    for true, pred, count in pairs[:15]:
        print(f"    {true} -> {pred}  x{count}")

    return results


if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHECKPOINT_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]
    labels, species_names = build_species_labels(paths)
    is_train = build_train_mask(paths)
    num_species = len(species_names)

    train_idx = is_train.nonzero(as_tuple=True)[0]
    val_idx = (~is_train).nonzero(as_tuple=True)[0]
    train_features = features[train_idx].to(device)
    val_features = features[val_idx].to(device)
    train_labels = labels[train_idx].to(device)
    val_labels = labels[val_idx].to(device)

    print(f"train: {train_features.shape[0]} images, val: {val_features.shape[0]} images, species: {num_species}")

    # raw-CLIP baseline: frozen features, no head, same protocol
    raw_train_emb = F.normalize(train_features, p=2, dim=1)
    raw_val_emb = F.normalize(val_features, p=2, dim=1)
    baseline_results = run_eval("raw-CLIP baseline", raw_train_emb, train_labels, raw_val_emb, val_labels, num_species, species_names)

    # trained head
    ckpt = load_checkpoint(checkpoint_path)
    head = ProjectionHead(feature_dim=ckpt["feature_dim"], embedding_dim=ckpt["embedding_dim"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    with torch.no_grad():
        head_train_emb = F.normalize(head(train_features), p=2, dim=1)
        head_val_emb = F.normalize(head(val_features), p=2, dim=1)
    head_results = run_eval(f"trained head ({checkpoint_path})", head_train_emb, train_labels, head_val_emb, val_labels, num_species, species_names)

    print("\n--- summary ---")
    print(f"{'k':>4} {'raw-CLIP':>10} {'trained head':>14} {'delta':>8}")
    for k in KS:
        delta = head_results[k] - baseline_results[k]
        print(f"{k:>4} {baseline_results[k]:>10.4f} {head_results[k]:>14.4f} {delta:>+8.4f}")
