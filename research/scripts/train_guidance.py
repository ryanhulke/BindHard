from pathlib import Path

import torch
from tqdm import tqdm

from common import (
    amp_settings,
    build_guidance_model,
    build_optimizer,
    load_config,
    move_batch_to_device,
    set_random_seed,
)
from config.config import GuidanceConfig
from model.sampling_guidance import PocketAffinityGuidance
from datamodules import build_datamodule

CONFIG_PATH = "config/train/guidance.yaml"


def compute_guidance_loss(
    model: PocketAffinityGuidance,
    batch: dict[str, torch.Tensor],
    *,
    target_name: str,
    loss_scale: float,
    guidance_loss: str = "mse",
    mask_positive_labels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_name not in batch:
        raise KeyError(f"guidance training requires {target_name!r} labels in the dataset")

    pred = model(
        batch=batch,
        protein_pos=batch["protein_pos"].float(),
        ligand_pos=batch["ligand_pos"].float(),
        ligand_type=batch["ligand_type"].long(),
    )
    target = batch[target_name].float().view_as(pred)
    loss_name = str(guidance_loss).lower()
    if loss_name == "mse":
        per_sample = (pred - target).pow(2)
    elif loss_name == "mae":
        per_sample = (pred - target).abs()
    else:
        raise ValueError(f"unsupported guidance_loss: {guidance_loss!r}")

    if mask_positive_labels:
        valid_mask = (target <= 0).to(per_sample.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        loss = (per_sample * valid_mask).sum() / denom
    else:
        loss = per_sample.mean()
    return loss * loss_scale, pred, target


@torch.no_grad()
def eval_epoch(
    model: PocketAffinityGuidance,
    loader,
    device: torch.device,
    *,
    target_name: str,
    guidance_loss: str = "mse",
    mask_positive_labels: bool = False,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, _, _ = compute_guidance_loss(
            model,
            batch,
            target_name=target_name,
            loss_scale=1.0,
            guidance_loss=guidance_loss,
            mask_positive_labels=mask_positive_labels,
        )
        bsz = int(batch["protein_counts"].numel())
        loss_sum += float(loss.item()) * bsz
        n += bsz

    return {"loss": loss_sum / max(1, n), "n": n}


def train_epoch(
    model: PocketAffinityGuidance,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    target_name: str,
    target_loss_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
    guidance_loss: str = "mse",
    mask_positive_labels: bool = False,
) -> dict[str, float | int]:
    model.train()
    loss_sum = 0.0
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bsz = int(batch["protein_counts"].numel())

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss, _, _ = compute_guidance_loss(
                model,
                batch,
                target_name=target_name,
                loss_scale=target_loss_scale,
                guidance_loss=guidance_loss,
                mask_positive_labels=mask_positive_labels,
            )

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        loss_sum += float(loss.item()) * bsz
        n += bsz

    return {"loss": loss_sum / max(1, n), "n": n}


def train() -> None:
    import wandb

    cfg = load_config(GuidanceConfig, CONFIG_PATH)
    set_random_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype = amp_settings(device, cfg.precision)

    dm = build_datamodule(
        cfg,
        guidance_label_path=cfg.guidance_label_path,
        guidance_target=cfg.guidance_target,
        drop_last=True,
    )
    if len(dm.ds_train) == 0:
        raise ValueError(
            f"no labeled training samples found for target {cfg.guidance_target!r} "
            f"in {cfg.guidance_label_path!r}"
        )
    if len(dm.ds_val) == 0:
        raise ValueError(
            f"no labeled validation samples found for target {cfg.guidance_target!r} "
            f"in {cfg.guidance_label_path!r}"
        )

    model = build_guidance_model(cfg, device)
    optimizer = build_optimizer(model, cfg, device)

    wandb.init(project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.__dict__)
    wandb.watch(model, log="gradients", log_freq=200)

    ckpt_dir = Path("checkpoints") / str(wandb.run.id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{cfg.run_name}_last.pt"
    best_path = ckpt_dir / f"{cfg.run_name}_best.pt"

    best_val = float("inf")
    seen = 0

    for epoch in range(cfg.epochs):
        train_loss = 0.0
        epoch_seen = 0
        for batch in tqdm(dm.train_dataloader(), desc=f"guidance epoch {epoch}"):
            batch = move_batch_to_device(batch, device)
            bsz = int(batch["protein_counts"].numel())

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, _, _ = compute_guidance_loss(
                    model,
                    batch,
                    target_name=cfg.guidance_target,
                    loss_scale=cfg.target_loss_scale,
                    guidance_loss=cfg.guidance_loss,
                    mask_positive_labels=cfg.mask_positive_guidance_labels,
                )

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            seen += bsz
            epoch_seen += bsz
            train_loss += float(loss.item()) * bsz
            wandb.log({"train/loss": float(loss.item()), "epoch": epoch, "seen": seen}, step=seen)

        val = eval_epoch(
            model,
            dm.val_dataloader(),
            device,
            target_name=cfg.guidance_target,
            guidance_loss=cfg.guidance_loss,
            mask_positive_labels=cfg.mask_positive_guidance_labels,
        )
        wandb.log(
            {
                "train/epoch_loss": train_loss / max(1, epoch_seen),
                "val/loss": val["loss"],
                "epoch": epoch,
            },
            step=seen,
        )

        payload = {
            "epoch": epoch,
            "seen": seen,
            "guidance": model.state_dict(),
            "opt": optimizer.state_dict(),
            "cfg": cfg.__dict__,
        }
        torch.save(payload, last_path)
        if val["loss"] < best_val:
            best_val = float(val["loss"])
            torch.save(payload, best_path)

    wandb.finish()


if __name__ == "__main__":
    train()
