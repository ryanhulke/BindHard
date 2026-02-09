from pathlib import Path
from dataclasses import dataclass
import yaml
import torch
import torch.nn as nn
import wandb

from datamodules import CrossDockedDataModule
from model.diffusion import LigandDiffusion
from model.egnn import EGNN


@dataclass
class TrainConfig:
    lmdb_path: str = "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
    split_path: str = "data/crossdocked_pose_split_from_name_val1000.pt"

    num_types: int = 7
    steps: int = 1000
    type_loss_scale: float = 100.0
    protein_noise_std: float = 0.1

    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 10

    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0

    log_every: int = 20
    val_every: int = 1
    precision: str = "bf16"  # bf16, fp16, fp32

    project: str = "bindhard-diffusion"
    run_name: str = "egnn_diffusion"


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def eval_epoch(diffusion: nn.Module, denoiser: nn.Module, loader, device: torch.device) -> dict:
    diffusion.eval()
    denoiser.eval()

    loss_sum = 0.0
    pos_sum = 0.0
    typ_sum = 0.0
    n = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        out = diffusion.loss(denoiser, batch)
        bsz = int(batch["protein_batch"].max().item()) + 1
        loss_sum += float(out["loss"].item()) * bsz
        pos_sum += float(out["loss_pos"].item()) * bsz
        typ_sum += float(out["loss_type"].item()) * bsz
        n += bsz

    return {
        "loss": loss_sum / max(1, n),
        "loss_pos": pos_sum / max(1, n),
        "loss_type": typ_sum / max(1, n),
    }


def train():
    with open("config/base_config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = TrainConfig(**cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = CrossDockedDataModule(
        lmdb_path=cfg.lmdb_path,
        split_pt_path=cfg.split_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    dm.setup()

    denoiser = EGNN(
        num_layers=6,
        hidden_dim=256,
        edge_feat_dim=4,
        num_r_gaussian=16,
        k=32,
        cutoff_mode="knn",
        update_x=True,
        act_fn="silu",
        norm=False,
    ).to(device)

    diffusion = LigandDiffusion(
        denoiser=denoiser,
        num_types=cfg.num_types,
        steps=cfg.steps,
        type_loss_scale=cfg.type_loss_scale,
        protein_noise_std=cfg.protein_noise_std,
    ).to(device)

    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    amp = cfg.precision.lower()
    use_amp = (device.type == "cuda") and (amp in {"bf16", "fp16"})
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    wandb.init(project=cfg.project, name=cfg.run_name, config=cfg.__dict__)
    wandb.watch(denoiser, log="gradients", log_freq=200)

    global_step = 0
    best_val = float("inf")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        diffusion.train()
        denoiser.train()

        for batch in dm.train_dataloader():
            batch = move_to_device(batch, device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = diffusion.loss(batch)
                loss = out["loss"]

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=cfg.grad_clip)
            opt.step()

            if global_step % cfg.log_every == 0:
                lr = float(opt.param_groups[0]["lr"])
                log = {
                    "train/loss": float(out["loss"].item()),
                    "train/loss_pos": float(out["loss_pos"].item()),
                    "train/loss_type": float(out["loss_type"].item()),
                    "train/lr": lr,
                    "epoch": epoch,
                    "step": global_step,
                }
                wandb.log(log, step=global_step)

            global_step += 1

        if (epoch + 1) % cfg.val_every == 0:
            val_metrics = eval_epoch(diffusion, denoiser, dm.val_dataloader(), device)
            wandb.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/loss_pos": val_metrics["loss_pos"],
                    "val/loss_type": val_metrics["loss_type"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                ckpt_path = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val": best_val,
                        "denoiser": denoiser.state_dict(),
                        "diffusion": diffusion.state_dict(),
                        "opt": opt.state_dict(),
                        "cfg": cfg.__dict__,
                    },
                    ckpt_path,
                )

        ckpt_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "denoiser": denoiser.state_dict(),
                "diffusion": diffusion.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg.__dict__,
            },
            ckpt_path,
        )

    wandb.finish()


if __name__ == "__main__":
    train()