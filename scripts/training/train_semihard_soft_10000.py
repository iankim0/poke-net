import torch

from model.train import TrainConfig, train

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = "cuda"

    cfg = TrainConfig(mining_strategy="semihard_soft")
    train(
        cfg,
        num_steps=10000,
        log_interval=100,
        device=device,
        log_dir="runs/semihard_soft_10000",
        checkpoint_path="checkpoints/semihard_soft_10000_head.pt",
        snapshot_interval=1000,
        snapshot_path_template="checkpoints/semihard_soft_snap{step}.pt",
    )
