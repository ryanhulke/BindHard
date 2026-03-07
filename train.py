from pathlib import Path
import yaml
import torch
import wandb
from tqdm import tqdm

from config.config import TrainConfig
from datamodules import CrossDockedDataModule
from model.flow_matching import LigandFlowMatching
from model.egnn import EGNN


@torch.no_grad()
def eval_epoch(diffusion: LigandFlowMatching, loader, device: torch.device) -> dict:
    diffusion.eval()
    loss_sum = 0.0
    pos_sum = 0.0
    typ_sum = 0.0
    n = 0

    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        out = diffusion.loss(batch)
        bsz = int(batch["protein_counts"].numel()) if "protein_counts" in batch else (int(batch["protein_batch"].max().item()) + 1)

        loss_sum += float(out["loss"].item()) * bsz
        pos_sum += float(out["loss_position"].item()) * bsz

        if "loss_atom_type_scaled" in out:
            typ_sum += float(out["loss_atom_type_scaled"].item()) * bsz
        else:
            typ_sum += float(out["loss_atom_type"].item()) * bsz

        n += bsz

    return {
        "loss": loss_sum / max(1, n),
        "loss_position": pos_sum / max(1, n),
        "loss_atom_type": typ_sum / max(1, n),
        "n": n,
    }


def train():
    with open("config/train/flow_matching.yaml", "r") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # if resuming training, these are needed
    resume_path = Path(cfg.resume_ckpt)
    ckpt = torch.load(resume_path, map_location="cpu") if (cfg.resume and resume_path.exists()) else None
    start_epoch = int(ckpt.get("epoch", -1)) + 1 if ckpt is not None else 0
    seen = int(ckpt.get("seen", 0)) if ckpt is not None else 0
    best_val = float(ckpt.get("best_val", float("inf"))) if ckpt is not None else float("inf")
    bad_epochs = int(ckpt.get("bad_epochs", 0)) if ckpt is not None else 0
    wandb_id = (ckpt.get("wandb_id", None) if ckpt is not None else None) if cfg.run_id is None else cfg.run_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = CrossDockedDataModule(
        lmdb_path=cfg.lmdb_path,
        split_pt_path=cfg.split_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor
    )

    denoiser = EGNN(
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        edge_feat_dim=cfg.edge_feat_dim,
        num_r_gaussian=cfg.num_r_gaussian,
        message_passing_mode=cfg.message_passing_mode,
        k=cfg.k,
        cutoff_mode=cfg.cutoff_mode,
        update_x=True,
        norm=cfg.norm,
    ).to(device)

    ligand_diffusion = LigandFlowMatching(
        denoiser=denoiser,
        num_types=cfg.num_types,
        steps=cfg.steps,
        type_loss_scale=cfg.type_loss_scale,
        protein_noise_std=cfg.protein_noise_std,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(ligand_diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)

    amp = cfg.precision.lower()
    use_amp = (device.type == "cuda") and (amp in {"bf16", "fp16"})
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    # load model weights & optimizer state if resuming training
    if ckpt is not None:
        ligand_diffusion.load_state_dict(ckpt["diffusion"], strict=True)
        optimizer.load_state_dict(ckpt["opt"])

    wandb_kwargs = dict(project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.__dict__)
    if wandb_id:
        wandb.init(**wandb_kwargs, id=wandb_id, resume="must")
    else:
        wandb.init(**wandb_kwargs)
        wandb_id = wandb.run.id

    ckpt_dir = Path("checkpoints") / str(wandb_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{cfg.run_name}_last.pt"
    best_path = ckpt_dir / f"{cfg.run_name}_best.pt"

    wandb.watch(ligand_diffusion, log="gradients", log_freq=200)


    ## TRAINING LOOP
    for epoch in range(start_epoch, cfg.epochs):
        ligand_diffusion.train()
        
        next_log_step = (seen // max(1, cfg.batch_size)) + cfg.log_every
        for batch in tqdm(dm.train_dataloader(), desc=f"epoch {epoch}"):
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            bsz = int(batch["protein_counts"].numel())

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = ligand_diffusion.loss(batch)
                loss = out["loss"]

            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ligand_diffusion.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            seen += bsz
            step_idx = seen // max(1, cfg.batch_size)

            if step_idx >= next_log_step:
                lt = out["loss_atom_type_scaled"] if "loss_atom_type_scaled" in out else out["loss_atom_type"]
                wandb.log(
                    {
                        "train/loss": float(out["loss"].item()),
                        "train/loss_position": float(out["loss_position"].item()),
                        "train/loss_atom_type": float(lt.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch": epoch,
                        "seen": seen,
                    },
                    step=int(seen),
                )
                next_log_step = step_idx + cfg.log_every

        if (epoch + 1) % cfg.val_every == 0:
            val = eval_epoch(ligand_diffusion, dm.val_dataloader(), device)
            wandb.log(
                {
                    "val/loss": val["loss"],
                    "val/loss_position": val["loss_position"],
                    "val/loss_atom_type": val["loss_atom_type"],
                    "epoch": epoch,
                },
                step=seen,
            )

            if val["loss"] < best_val:
                best_val = val["loss"]
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "seen": seen,
                        "best_val": best_val,
                        "bad_epochs": bad_epochs,
                        "wandb_id": wandb_id,
                        "diffusion": ligand_diffusion.state_dict(),
                        "opt": optimizer.state_dict(),
                        "cfg": cfg.__dict__,
                    },
                    best_path,
                )
            else:
                bad_epochs += 1

            stop = cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience

        torch.save(
            {
                "epoch": epoch,
                "seen": seen,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "wandb_id": wandb_id,
                "diffusion": ligand_diffusion.state_dict(),
                "opt": optimizer.state_dict(),
                "cfg": cfg.__dict__,
            },
            last_path,
        )

        if stop:
            break

    wandb.finish()


if __name__ == "__main__":
    train()
