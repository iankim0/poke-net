from pathlib import Path

import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}


def collect_image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def build_feature_cache(encoder, root: Path, device, batch_size: int = 64) -> dict:
    """Run every image under root through encoder once; return a dict ready for torch.save."""
    paths = collect_image_paths(root)
    relative_paths = [str(p.relative_to(root)).replace("\\", "/") for p in paths]

    encoder = encoder.to(device)
    features = torch.empty(len(paths), encoder.feature_dim)

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [encoder.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(images).to(device)
        feats = encoder(batch)
        features[start : start + len(batch_paths)] = feats.cpu()

    return {
        "paths": relative_paths,
        "features": features,
        "model_name": encoder.model_name,
        "pretrained": encoder.pretrained,
        "feature_dim": encoder.feature_dim,
    }


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_cache(path: Path) -> dict:
    return torch.load(path)


def as_lookup(cache: dict) -> dict[str, torch.Tensor]:
    """Convert the stored (paths, features) arrays into a {relative_path: feature} dict."""
    return dict(zip(cache["paths"], cache["features"]))
