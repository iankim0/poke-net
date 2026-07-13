import torch

from model.train import TrainConfig, train

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = "cuda"

    cfg = TrainConfig(mining_strategy="semihard_soft")
    train(
        cfg,
        num_steps=1500,
        log_interval=10,
        device=device,
        log_dir="runs/semihard_soft_1500",
        checkpoint_path="checkpoints/semihard_soft_1500_head.pt",
    )
