import torch
import torch.nn.functional as F

from model.baseline import cosine_topk_accuracy
from model.dataset import build_species_labels
from model.feature_cache import load_cache
from model.train import TrainConfig, train

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
K = 5

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = "cuda"

    cfg = TrainConfig(mining_strategy="semihard")
    head = train(
        cfg,
        num_steps=500,
        log_interval=10,
        device=device,
        log_dir="runs/full_semihard",
        checkpoint_path="checkpoints/full_semihard_head.pt",
    )

    # full real-image top-5 eval, same method as the raw-CLIP baseline, for direct comparison
    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]

    real_mask = torch.tensor([p.split("/", 1)[1].startswith("real_") for p in paths])
    real_paths = [p for p, keep in zip(paths, real_mask.tolist()) if keep]
    real_features = features[real_mask].to(device)

    labels, species_names = build_species_labels(real_paths)
    labels = labels.to(device)

    head.eval()
    with torch.no_grad():
        embeddings = F.normalize(head(real_features), p=2, dim=1)
    accuracy = cosine_topk_accuracy(embeddings, labels, k=K)

    print(f"\nfull-dataset trained-head top-{K} accuracy (precision@{K}): {accuracy:.4f}")
    print("raw-CLIP baseline (reference): 0.5359")
    print("soft-margin full run (reference): 0.5584")
