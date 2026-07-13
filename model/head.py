import torch.nn as nn


class ProjectionHead(nn.Module):
    """Maps a backbone feature vector to the shared embedding space."""

    def __init__(self, feature_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)  # L2-normalize happens in EmbeddingModel
