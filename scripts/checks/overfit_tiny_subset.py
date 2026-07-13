import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model.baseline import cosine_topk_accuracy
from model.dataset import build_species_labels
from model.feature_cache import load_cache
from model.head import ProjectionHead
from model.pk_sampler import PKSampler
from model.tiny_subset import build_tiny_subset
from model.triplet import batch_triplet_loss, embedding_spread

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
TINY_SPECIES = ["025_pikachu", "006_charizard", "094_gengar", "143_snorlax", "133_eevee"]
IMAGES_PER_SPECIES = 10

MARGIN = 0.2
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_STEPS = 500
LOG_INTERVAL = 10
SEED = 0

RAW_CLIP_BASELINE = 0.5359  # full-dataset top-5, for reference


def run_overfit(strategy: str, subset_features: torch.Tensor, labels: torch.Tensor, species_names: list, device: str):
    torch.manual_seed(SEED)

    P, K = len(species_names), IMAGES_PER_SPECIES
    sampler = PKSampler(labels.cpu(), P=P, K=K, num_batches=NUM_STEPS)

    head = ProjectionHead(feature_dim=subset_features.shape[1], embedding_dim=128).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(f"runs/overfit_tiny_subset_{strategy}")

    print(f"\n=== strategy={strategy} ===")
    for step, batch_local in enumerate(sampler):
        batch_local_t = torch.tensor(batch_local, device=device)
        batch_features = subset_features[batch_local_t]
        batch_labels = labels[batch_local_t]

        embeddings = F.normalize(head(batch_features), p=2, dim=1)
        loss, stats = batch_triplet_loss(embeddings, batch_labels, margin=MARGIN, strategy=strategy)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % LOG_INTERVAL == 0:
            with torch.no_grad():
                spread = embedding_spread(embeddings)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/frac_active", stats["frac_active"], step)
            writer.add_scalar("train/mean_pos_dist", stats["mean_pos_dist"], step)
            writer.add_scalar("train/mean_neg_dist", stats["mean_neg_dist"], step)
            writer.add_scalar("train/embedding_spread", spread, step)
            print(f"step {step}: loss={loss.item():.4f} spread={spread:.4f} stats={stats}")

    writer.close()

    head.eval()
    with torch.no_grad():
        final_embeddings = F.normalize(head(subset_features), p=2, dim=1)
    subset_topk = cosine_topk_accuracy(final_embeddings, labels, k=5)

    return {
        "final_loss": loss.item(),
        "final_frac_active": stats["frac_active"],
        "final_mean_pos_dist": stats["mean_pos_dist"],
        "final_mean_neg_dist": stats["mean_neg_dist"],
        "subset_top5": subset_topk,
    }


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = "cuda"

    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]

    subset_indices = build_tiny_subset(paths, TINY_SPECIES, images_per_species=IMAGES_PER_SPECIES)
    subset_paths = [paths[i] for i in subset_indices]
    subset_features = features[subset_indices].to(device)

    labels, species_names = build_species_labels(subset_paths)
    labels = labels.to(device)

    print(f"tiny subset: {len(subset_paths)} images, {len(species_names)} species: {species_names}")

    results = {}
    for strategy in ["hard", "soft"]:
        results[strategy] = run_overfit(strategy, subset_features, labels, species_names, device)

    print("\n=== head-to-head comparison ===")
    print(f"raw-CLIP full-dataset baseline (reference): {RAW_CLIP_BASELINE:.4f}")
    for strategy, r in results.items():
        print(
            f"{strategy:5s}: final_loss={r['final_loss']:.4f} frac_active={r['final_frac_active']:.4f} "
            f"pos_dist={r['final_mean_pos_dist']:.4f} neg_dist={r['final_mean_neg_dist']:.4f} "
            f"subset_top5={r['subset_top5']:.4f}"
        )
