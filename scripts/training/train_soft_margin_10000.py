import torch

from model.train import TrainConfig, train

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = "cuda"

    cfg = TrainConfig(mining_strategy="soft")
    train(
        cfg,
        num_steps=10000,
        log_interval=500,
        device=device,
        log_dir="runs/soft_margin_10000",
        checkpoint_path="checkpoints/soft_margin_10000_head.pt",
    )
