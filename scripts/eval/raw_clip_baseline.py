import torch

from model.baseline import cosine_topk_accuracy
from model.dataset import build_species_labels
from model.feature_cache import load_cache

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
K = 5

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]

    real_mask = torch.tensor([p.split("/", 1)[1].startswith("real_") for p in paths])
    real_paths = [p for p, keep in zip(paths, real_mask.tolist()) if keep]
    real_features = features[real_mask].to(device)

    labels, species_names = build_species_labels(real_paths)
    labels = labels.to(device)

    accuracy = cosine_topk_accuracy(real_features, labels, k=K)

    print(f"images: {len(real_paths)}, species: {len(species_names)}")
    print(f"raw-CLIP top-{K} accuracy (precision@{K}): {accuracy:.4f}")
