from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from common import (
    amp_settings,
    build_datamodule,
    build_diffusion_model,
    build_optimizer,
    load_config,
    move_batch_to_device,
    resolve_path,
    set_random_seed,
)
from config.config import TrainConfig


CONFIG_PATH = "config/train/flow_matching.yaml"


@torch.no_grad()
def eval_epoch(model, loader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    stats = {
        "loss": 0.0,
        "loss_position": 0.0,
        "loss_atom_type": 0.0,
        "loss_bond": 0.0,
        "loss_charge": 0.0,
        "loss_atom_count": 0.0,
        "n": 0,
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model.loss(batch)
        bsz = int(batch["protein_counts"].numel())

        stats["loss"] += float(out["loss"].item()) * bsz
        stats["loss_position"] += float(out["loss_position"].item()) * bsz
        stats["loss_atom_type"] += float(
            out.get("loss_atom_type_scaled", out["loss_atom_type"]).item()
        ) * bsz
        stats["loss_bond"] += float(out.get("loss_bond", torch.zeros(())).item()) * bsz
        stats["loss_charge"] += float(out.get("loss_charge", torch.zeros(())).item()) * bsz
        stats["loss_atom_count"] += float(out.get("loss_atom_count", torch.zeros(())).item()) * bsz
        stats["n"] += bsz

    n = max(1, int(stats["n"]))
    return {key: value / n if key != "n" else value for key, value in stats.items()}


def train() -> None:
    cfg = load_config(TrainConfig, CONFIG_PATH)
    set_random_seed(cfg.seed)

    # if resuming training, these are needed
    resume_path = resolve_path(cfg.resume_ckpt)
    ckpt = torch.load(resume_path, map_location="cpu") if cfg.resume and resume_path.exists() else None
    start_epoch = int(ckpt.get("epoch", -1)) + 1 if ckpt is not None else 0
    seen = int(ckpt.get("seen", 0)) if ckpt is not None else 0
    best_val = float(ckpt.get("best_val", float("inf"))) if ckpt is not None else float("inf")
    bad_epochs = int(ckpt.get("bad_epochs", 0)) if ckpt is not None else 0
    wandb_id = cfg.run_id or (ckpt.get("wandb_id") if ckpt is not None else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype = amp_settings(device, cfg.precision)

    dm = build_datamodule(cfg)
    model = build_diffusion_model(cfg, device)
    optimizer = build_optimizer(model, cfg, device)

    # load model weights & optimizer state if resuming training
    if ckpt is not None:
        model.load_state_dict(ckpt["diffusion"], strict=False)
        optimizer.load_state_dict(ckpt["opt"])

    wandb_kwargs = {
        "project": cfg.project,
        "entity": cfg.entity,
        "name": cfg.run_name,
        "config": cfg.__dict__,
    }
    if wandb_id:
        wandb.init(**wandb_kwargs, id=wandb_id, resume="must")
    else:
        wandb.init(**wandb_kwargs)
        wandb_id = wandb.run.id

    ckpt_dir = Path("checkpoints") / str(wandb_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{cfg.run_name}_last.pt"
    best_path = ckpt_dir / f"{cfg.run_name}_best.pt"

    wandb.watch(model, log="gradients", log_freq=200)
    stop = False

    ## TRAINING LOOP
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        next_log_step = (seen // max(1, cfg.batch_size)) + cfg.log_every

        for batch in tqdm(dm.train_dataloader(), desc=f"epoch {epoch}"):
            batch = move_batch_to_device(batch, device)
            bsz = int(batch["protein_counts"].numel())

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model.loss(batch)
                loss = out["loss"]

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            seen += bsz
            step_idx = seen // max(1, cfg.batch_size)
            if step_idx < next_log_step:
                continue

            wandb.log(
                {
                    "train/loss": float(out["loss"].item()),
                    "train/loss_position": float(out["loss_position"].item()),
                    "train/loss_atom_type": float(
                        out.get("loss_atom_type_scaled", out["loss_atom_type"]).item()
                    ),
                    "train/loss_atom_count": float(out.get("loss_atom_count", torch.zeros(())).item()),
                    "train/loss_charge": float(out.get("loss_charge", torch.zeros(())).item()),
                    "train/loss_bond": float(out.get("loss_bond", torch.zeros(())).item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "epoch": epoch,
                    "seen": seen,
                },
                step=seen,
            )
            next_log_step = step_idx + cfg.log_every

        if (epoch + 1) % cfg.val_every == 0:
            val = eval_epoch(model, dm.val_dataloader(), device)
            wandb.log(
                {
                    "val/loss": val["loss"],
                    "val/loss_position": val["loss_position"],
                    "val/loss_atom_type": val["loss_atom_type"],
                    "val/loss_atom_count": val["loss_atom_count"],
                    "val/loss_charge": val["loss_charge"],
                    "val/loss_bond": val["loss_bond"],
                    "epoch": epoch,
                },
                step=seen,
            )

            is_best = val["loss"] < best_val
            if is_best:
                best_val = float(val["loss"])
                bad_epochs = 0
            else:
                bad_epochs += 1

            payload = {
                "epoch": epoch,
                "seen": seen,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "wandb_id": wandb_id,
                "diffusion": model.state_dict(),
                "opt": optimizer.state_dict(),
                "cfg": cfg.__dict__,
            }
            if is_best:
                torch.save(payload, best_path)
            stop = cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience
        else:
            payload = {
                "epoch": epoch,
                "seen": seen,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "wandb_id": wandb_id,
                "diffusion": model.state_dict(),
                "opt": optimizer.state_dict(),
                "cfg": cfg.__dict__,
            }

        torch.save(payload, last_path)
        if stop:
            break

    wandb.finish()


if __name__ == "__main__":
    train()
