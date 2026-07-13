import torch
import torch.nn as nn
import open_clip


class FrozenCLIPEncoder(nn.Module):
    """Frozen OpenCLIP image encoder. forward(x) -> (B, feature_dim)."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        super().__init__()
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model = clip_model
        self.feature_dim = clip_model.visual.output_dim
        self.model_name = model_name
        self.pretrained = pretrained

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

    def train(self, mode: bool = True):
        # Never leave eval mode, even if a parent module's .train() cascades here.
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.clip_model.encode_image(x)
