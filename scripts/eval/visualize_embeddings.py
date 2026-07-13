from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from model.dataset import build_species_labels, build_train_mask
from model.feature_cache import load_cache
from model.head import ProjectionHead
from model.train import load_checkpoint

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
CHECKPOINT_PATH = "checkpoints/semihard_soft_10000_head.pt"
OUT_PATH = "plots/tsne.png"
PERPLEXITY = 30
SEED = 0

# a legible subset: four evolution lines (the hard cases from the confusion breakdown) plus a
# few visually-distinct control species that should stay cleanly separated.
SPECIES_SUBSET = [
    "bulbasaur", "ivysaur", "venusaur",
    "charmander", "charmeleon", "charizard",
    "squirtle", "wartortle", "blastoise",
    "pidgey", "pidgeotto", "pidgeot",
    "zubat", "golbat",
    "drowzee", "hypno",
    "snorlax", "gengar", "mewtwo",
]


def select_species(species_names: list[str], keywords: list[str]) -> list[str]:
    selected = []
    for kw in keywords:
        matches = [s for s in species_names if s.split("_", 1)[1] == kw]
        if not matches:
            raise ValueError(f"no species matched keyword {kw!r}")
        selected.append(matches[0])
    return selected


def run_tsne(embeddings: torch.Tensor) -> "torch.Tensor":
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=SEED, init="pca")
    return tsne.fit_transform(embeddings.cpu().numpy())


def plot_panel(ax, coords, subset_labels, is_val, subset_names, label_to_name, colors, title):
    for species_name, color in zip(subset_names, colors):
        species_idx = [i for i, n in label_to_name.items() if n == species_name][0]
        mask = subset_labels == species_idx

        train_mask = mask & ~is_val
        val_mask = mask & is_val

        ax.scatter(coords[train_mask, 0], coords[train_mask, 1], color=color, s=14, alpha=0.5, label=species_name)
        ax.scatter(coords[val_mask, 0], coords[val_mask, 1], color=color, s=140, marker="*",
                   edgecolor="black", linewidth=0.8, zorder=5)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]
    labels, species_names = build_species_labels(paths)
    is_train = build_train_mask(paths)
    label_to_name = dict(enumerate(species_names))

    subset_names = select_species(species_names, SPECIES_SUBSET)
    subset_label_ids = {i for i, n in label_to_name.items() if n in subset_names}
    subset_mask = torch.tensor([label.item() in subset_label_ids for label in labels])

    subset_features = features[subset_mask].to(device)
    subset_labels = labels[subset_mask]
    subset_is_val = ~is_train[subset_mask]

    print(f"subset: {len(subset_names)} species, {subset_features.shape[0]} images "
          f"({subset_is_val.sum().item()} val)")

    raw_emb = F.normalize(subset_features, p=2, dim=1)

    ckpt = load_checkpoint(CHECKPOINT_PATH)
    head = ProjectionHead(feature_dim=ckpt["feature_dim"], embedding_dim=ckpt["embedding_dim"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    with torch.no_grad():
        head_emb = F.normalize(head(subset_features), p=2, dim=1)

    raw_coords = run_tsne(raw_emb)
    head_coords = run_tsne(head_emb)

    colors = plt.cm.tab20(range(len(subset_names)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    plot_panel(axes[0], raw_coords, subset_labels, subset_is_val, subset_names, label_to_name, colors,
               "raw-CLIP (no head)")
    plot_panel(axes[1], head_coords, subset_labels, subset_is_val, subset_names, label_to_name, colors,
               f"trained head ({CHECKPOINT_PATH})")

    handles, plot_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc="center right", fontsize=9, markerscale=1.5)
    fig.suptitle("t-SNE of embeddings: train (small dots) + val (large stars), colored by species")
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved -> {OUT_PATH}")
