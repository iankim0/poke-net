import torch
import torch.nn.functional as F
from PIL import Image

from model.encoder import FrozenCLIPEncoder
from model.eval import species_scores
from model.head import ProjectionHead


def load_bundle(bundle_path: str) -> dict:
    return torch.load(bundle_path)


def build_model(bundle: dict) -> tuple[FrozenCLIPEncoder, ProjectionHead]:
    """CPU-only: a single-image forward pass is cheap enough that CUDA is never required here."""
    encoder = FrozenCLIPEncoder(bundle["backbone_model_name"], bundle["backbone_pretrained"])
    head = ProjectionHead(feature_dim=bundle["feature_dim"], embedding_dim=bundle["embedding_dim"])
    head.load_state_dict(bundle["head_state_dict"])
    head.eval()
    return encoder, head


def predict(image: Image.Image, encoder: FrozenCLIPEncoder, head: ProjectionHead, bundle: dict,
            k: int = 3) -> list[dict]:
    """Embeds one image and ranks it against the bundle's reference embeddings. Returns the top-k
    as a list of {species, score, gap_to_next} dicts, best first. gap_to_next is the similarity
    margin over the next-ranked species (None for the last entry)."""
    tensor = encoder.preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = F.normalize(head(encoder(tensor)), p=2, dim=1)

    sims = embedding @ bundle["reference_embeddings"].t()
    scores = species_scores(sims, bundle["reference_labels"], num_species=len(bundle["species_names"]))[0]
    ranked_scores, ranked_idx = scores.sort(descending=True)

    results = []
    for i in range(k):
        gap = (ranked_scores[i] - ranked_scores[i + 1]).item() if i + 1 < len(ranked_scores) else None
        results.append({
            "species": bundle["species_names"][ranked_idx[i].item()],
            "score": ranked_scores[i].item(),
            "gap_to_next": gap,
        })
    return results
