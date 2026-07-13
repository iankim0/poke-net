from pathlib import Path

import torch
from PIL import Image

from model.encoder import FrozenCLIPEncoder
from model.embedding_model import EmbeddingModel

SAMPLE_IMAGES = [
    "images/001_bulbasaur/real_0001.jpg",
    "images/004_charmander/real_0001.jpg",
    "images/007_squirtle/real_0001.jpg",
    "images/025_pikachu/real_0001.jpg",
]

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this check"
    device = "cuda"

    encoder = FrozenCLIPEncoder()
    model = EmbeddingModel(encoder, embedding_dim=128).to(device)

    batch = torch.stack(
        [encoder.preprocess(Image.open(p).convert("RGB")) for p in SAMPLE_IMAGES]
    ).to(device)

    output = model(batch)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

    print("device:", output.device)
    print("output shape:", tuple(output.shape))
    print("per-row L2 norms:", output.norm(dim=1))
    print("encoder trainable params:", n_encoder_trainable)
    print("head trainable params:", n_head_trainable)
    print("total trainable params:", n_trainable)
