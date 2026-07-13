import sys

import torch
import torch.nn.functional as F

from model.dataset import build_species_labels, build_train_mask
from model.eval import species_ranking, topk_accuracy
from model.feature_cache import load_cache
from model.head import ProjectionHead
from model.references import build_references
from model.train import load_checkpoint

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
NUM_REFS_PER_SPECIES = 5
KS = (1, 5, 10)


def eval_embeddings(train_embeddings, train_labels, val_embeddings, val_labels, num_species) -> dict:
    ref_emb, ref_labels = build_references(train_embeddings, train_labels, num_species=num_species, k=NUM_REFS_PER_SPECIES)
    sims = val_embeddings @ ref_emb.t()
    ranked = species_ranking(sims, ref_labels, num_species=num_species)
    return {k: topk_accuracy(ranked, val_labels, k=k) for k in KS}


if __name__ == "__main__":
    checkpoint_paths = sys.argv[1:]
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

    print(f"train: {train_features.shape[0]} images, val: {val_features.shape[0]} images, species: {num_species}\n")

    raw_train_emb = F.normalize(train_features, p=2, dim=1)
    raw_val_emb = F.normalize(val_features, p=2, dim=1)
    baseline = eval_embeddings(raw_train_emb, train_labels, raw_val_emb, val_labels, num_species)

    rows = [("raw-CLIP baseline", baseline)]

    for checkpoint_path in checkpoint_paths:
        ckpt = load_checkpoint(checkpoint_path)
        head = ProjectionHead(feature_dim=ckpt["feature_dim"], embedding_dim=ckpt["embedding_dim"]).to(device)
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()

        with torch.no_grad():
            head_train_emb = F.normalize(head(train_features), p=2, dim=1)
            head_val_emb = F.normalize(head(val_features), p=2, dim=1)
        results = eval_embeddings(head_train_emb, train_labels, head_val_emb, val_labels, num_species)
        rows.append((f"step={ckpt['step']} ({checkpoint_path})", results))

    print(f"{'checkpoint':>55} {'top-1':>8} {'top-5':>8} {'top-10':>8}")
    for name, results in rows:
        print(f"{name:>55} {results[1]:>8.4f} {results[5]:>8.4f} {results[10]:>8.4f}")
