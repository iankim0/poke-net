import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model.dataset import build_species_labels, build_train_mask
from model.feature_cache import load_cache
from model.head import ProjectionHead
from model.pk_sampler import PKSampler
from model.triplet import batch_triplet_loss, embedding_spread


@dataclass
class TrainConfig:
    cache_path: str = "cache/clip_vitb32_laion2b_features.pt"
    embedding_dim: int = 128

    P: int = 16
    K: int = 4
    margin: float = 0.2
    mining_strategy: str = "hard"

    lr: float = 1e-4
    weight_decay: float = 1e-4

    seed: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(head: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def save_checkpoint(path: str, head: torch.nn.Module, cache: dict, cfg: TrainConfig, step: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "backbone_model_name": cache["model_name"],
            "backbone_pretrained": cache["pretrained"],
            "feature_dim": cache["feature_dim"],
            "embedding_dim": cfg.embedding_dim,
            "margin": cfg.margin,
            "P": cfg.P,
            "K": cfg.K,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: str) -> dict:
    return torch.load(path)


def train(
    cfg: TrainConfig,
    num_steps: int,
    log_interval: int,
    device: str,
    log_dir: str = None,
    checkpoint_path: str = None,
    snapshot_interval: int = None,
    snapshot_path_template: str = None,
):
    """snapshot_interval/snapshot_path_template: if both given, save a checkpoint every
    `snapshot_interval` steps to snapshot_path_template.format(step=step) — lets a single training
    run produce a step-vs-accuracy curve without restarting from scratch at each length."""
    set_seed(cfg.seed)

    cache = load_cache(cfg.cache_path)
    paths, features = cache["paths"], cache["features"]
    labels, species_names = build_species_labels(paths)
    is_train = build_train_mask(paths)

    train_indices = is_train.nonzero(as_tuple=True)[0]
    train_labels = labels[train_indices]

    sampler = PKSampler(train_labels, P=cfg.P, K=cfg.K, num_batches=num_steps)

    head = ProjectionHead(feature_dim=cache["feature_dim"], embedding_dim=cfg.embedding_dim).to(device)
    optimizer = build_optimizer(head, cfg)
    writer = SummaryWriter(log_dir)

    for step, batch_local in enumerate(sampler):
        batch_local_t = torch.tensor(batch_local)
        global_indices = train_indices[batch_local_t]

        batch_features = features[global_indices].to(device)
        batch_labels = train_labels[batch_local_t].to(device)

        embeddings = F.normalize(head(batch_features), p=2, dim=1)
        loss, stats = batch_triplet_loss(embeddings, batch_labels, margin=cfg.margin, strategy=cfg.mining_strategy)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_interval == 0:
            with torch.no_grad():
                spread = embedding_spread(embeddings)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/frac_active", stats["frac_active"], step)
            writer.add_scalar("train/mean_pos_dist", stats["mean_pos_dist"], step)
            writer.add_scalar("train/mean_neg_dist", stats["mean_neg_dist"], step)
            writer.add_scalar("train/embedding_spread", spread, step)
            print(f"step {step}: loss={loss.item():.4f} spread={spread:.4f} stats={stats}")

        if snapshot_interval is not None and (step + 1) % snapshot_interval == 0:
            save_checkpoint(snapshot_path_template.format(step=step + 1), head, cache, cfg, step=step + 1)

    writer.close()

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, head, cache, cfg, step=num_steps)

    return head


def verify_checkpoint_roundtrip(checkpoint_path: str, trained_head: torch.nn.Module, device: str) -> None:
    ckpt = load_checkpoint(checkpoint_path)

    reloaded_head = ProjectionHead(feature_dim=ckpt["feature_dim"], embedding_dim=ckpt["embedding_dim"]).to(device)
    reloaded_head.load_state_dict(ckpt["head_state_dict"])

    trained_sd = trained_head.state_dict()
    reloaded_sd = reloaded_head.state_dict()
    for key in trained_sd:
        assert torch.equal(trained_sd[key], reloaded_sd[key]), f"mismatch in {key}"

    dummy = torch.randn(4, ckpt["feature_dim"], device=device)
    trained_head.eval()
    reloaded_head.eval()
    with torch.no_grad():
        out_trained = trained_head(dummy)
        out_reloaded = reloaded_head(dummy)
    assert torch.equal(out_trained, out_reloaded), "forward pass mismatch after reload"

    print(f"checkpoint round-trip OK: step={ckpt['step']}, backbone={ckpt['backbone_model_name']}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this run"
    device = "cuda"

    cfg = TrainConfig()
    trained_head = train(
        cfg,
        num_steps=500,
        log_interval=10,
        device=device,
        log_dir="runs/p16k4_lr1e-4",
        checkpoint_path="checkpoints/p16k4_lr1e-4_head.pt",
    )
    verify_checkpoint_roundtrip("checkpoints/p16k4_lr1e-4_head.pt", trained_head, device)
