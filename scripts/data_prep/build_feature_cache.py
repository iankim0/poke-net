import time
from pathlib import Path

import torch

from model.encoder import FrozenCLIPEncoder
from model.feature_cache import build_feature_cache, save_cache

IMAGES_ROOT = Path("images")
CACHE_PATH = Path("cache/clip_vitb32_laion2b_features.pt")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FrozenCLIPEncoder()

    start = time.time()
    cache = build_feature_cache(encoder, IMAGES_ROOT, device)
    elapsed = time.time() - start

    save_cache(cache, CACHE_PATH)

    print(f"Cached {len(cache['paths'])} images in {elapsed:.1f}s -> {CACHE_PATH}")
    print(f"Feature tensor shape: {cache['features'].shape}")
