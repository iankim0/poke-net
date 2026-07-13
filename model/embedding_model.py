import torch.nn as nn
import torch.nn.functional as F

from model.head import ProjectionHead


class EmbeddingModel(nn.Module):
    """encoder -> projection head -> L2-normalize. forward(x) -> (B, embedding_dim)."""

    def __init__(self, encoder: nn.Module, embedding_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = ProjectionHead(feature_dim=encoder.feature_dim, embedding_dim=embedding_dim)

    def forward(self, x):
        features = self.encoder(x)
        embedding = self.head(features)
        return F.normalize(embedding, p=2, dim=1)
