from pathlib import Path

import torch
import torch.nn.functional as F

from model.dataset import build_species_labels
from model.feature_cache import load_cache
from model.head import ProjectionHead
from model.references import build_references
from model.train import load_checkpoint

CACHE_PATH = "cache/clip_vitb32_laion2b_features.pt"
CHECKPOINT_PATH = "checkpoints/semihard_soft_10000_head.pt"
NUM_REFS_PER_SPECIES = 5
OUT_PATH = "export/pokemon_species_id_bundle.pt"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # every image, train and val alike — there's no more accuracy comparison to protect once this
    # bundle is the shipped artifact, so val no longer needs to be held out.
    cache = load_cache(CACHE_PATH)
    paths, features = cache["paths"], cache["features"]
    labels, species_names = build_species_labels(paths)
    num_species = len(species_names)

    features = features.to(device)
    labels = labels.to(device)

    ckpt = load_checkpoint(CHECKPOINT_PATH)
    head = ProjectionHead(feature_dim=ckpt["feature_dim"], embedding_dim=ckpt["embedding_dim"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    with torch.no_grad():
        embeddings = F.normalize(head(features), p=2, dim=1)

    ref_embeddings, ref_labels = build_references(embeddings, labels, num_species=num_species, k=NUM_REFS_PER_SPECIES)

    expected_refs = num_species * NUM_REFS_PER_SPECIES
    assert ref_embeddings.shape[0] <= expected_refs, "more references than species*k — build_references bug"
    assert torch.allclose(ref_embeddings.norm(dim=1), torch.ones(ref_embeddings.shape[0], device=device), atol=1e-4), \
        "reference embeddings aren't unit-norm"

    bundle = {
        "head_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
        "backbone_model_name": cache["model_name"],
        "backbone_pretrained": cache["pretrained"],
        "feature_dim": ckpt["feature_dim"],
        "embedding_dim": ckpt["embedding_dim"],
        "reference_embeddings": ref_embeddings.cpu(),
        "reference_labels": ref_labels.cpu(),
        "species_names": species_names,
        "num_refs_per_species": NUM_REFS_PER_SPECIES,
        "source_checkpoint": CHECKPOINT_PATH,
        "num_images_used": features.shape[0],
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, OUT_PATH)

    print(f"exported bundle -> {OUT_PATH}")
    print(f"  backbone: {cache['model_name']} ({cache['pretrained']})")
    print(f"  species: {num_species}, images used for references: {features.shape[0]}")
    print(f"  reference vectors: {ref_embeddings.shape[0]} (up to {NUM_REFS_PER_SPECIES}/species)")
