from __future__ import annotations

import random
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import wandb
from tqdm import tqdm

from common import (
    VinaPoseScorer,
    amp_settings,
    build_atom_type_decoder,
    build_diffusion_model,
    build_optimizer,
    compute_vina_sa_reward,
    load_config,
    move_batch_to_device,
    resolve_path,
    set_random_seed,
)
from datamodules import build_datamodule
from config.config import OnlineRLConfig


CONFIG_PATH = "config/train/rl_vina_sa.yaml"
PROTEIN_KEYS = (
    "protein_pos",
    "protein_element",
    "protein_atom_to_aa_type",
    "protein_is_backbone",
    "protein_batch",
)


def collate_rollout_batch(records: list[dict]) -> dict[str, torch.Tensor]:
    protein_pos = []
    protein_element = []
    protein_aa = []
    protein_bb = []
    protein_batch = []

    ligand_pos = []
    ligand_type = []
    ligand_batch = []
    reward = []

    for batch_idx, record in enumerate(records):
        p_count = int(record["protein_pos"].shape[0])
        l_count = int(record["ligand_pos"].shape[0])

        protein_pos.append(record["protein_pos"])
        protein_element.append(record["protein_element"])
        protein_aa.append(record["protein_atom_to_aa_type"])
        protein_bb.append(record["protein_is_backbone"])
        protein_batch.append(torch.full((p_count,), batch_idx, dtype=torch.long))

        ligand_pos.append(record["ligand_pos"])
        ligand_type.append(record["ligand_type"])
        ligand_batch.append(torch.full((l_count,), batch_idx, dtype=torch.long))
        reward.append(float(record["reward"]))

    return {
        "protein_pos": torch.cat(protein_pos, dim=0),
        "protein_element": torch.cat(protein_element, dim=0),
        "protein_atom_to_aa_type": torch.cat(protein_aa, dim=0),
        "protein_is_backbone": torch.cat(protein_bb, dim=0),
        "protein_batch": torch.cat(protein_batch, dim=0),
        "ligand_pos": torch.cat(ligand_pos, dim=0),
        "ligand_type": torch.cat(ligand_type, dim=0),
        "ligand_batch": torch.cat(ligand_batch, dim=0),
        "reward": torch.tensor(reward, dtype=torch.float32),
    }


@torch.no_grad()
def collect_rollouts(
    anchor_model,
    batch_cpu: dict[str, torch.Tensor | list],
    batch_gpu: dict[str, torch.Tensor | list],
    atom_type_decoder: dict[int, int],
    cfg: OnlineRLConfig,
    reward_dir: Path,
    reward_ema_mean: float | None,
    reward_ema_var: float | None,
) -> tuple[list[dict], dict[str, float | int], float | None, float | None]:
    protein_batch_gpu = {key: batch_gpu[key] for key in PROTEIN_KEYS}
    sampled = anchor_model.sample(
        protein_batch_gpu,
        prior=None,
        num_samples=cfg.rollout_num_samples,
        num_steps=cfg.rollout_steps,
        time_schedule=cfg.rollout_time_schedule,
        type_temperature=cfg.rollout_type_temperature,
    )
    sampled_cpu = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in sampled.items()
        if key != "trajectory"
    }

    ligand_batch_all = sampled_cpu["ligand_batch"]
    bond_index_all = sampled_cpu["ligand_bond_index"]
    bond_type_all = sampled_cpu["ligand_bond_type"]
    batch_size = int(batch_cpu["protein_counts"].numel())

    collected = []
    all_raw_rewards: list[float] = []
    invalid_count = 0
    total_count = 0

    for pocket_idx in range(batch_size):
        protein_mask = batch_cpu["protein_batch"] == pocket_idx
        ref_ligand_mask = batch_cpu["ligand_batch"] == pocket_idx
        protein = {
            "protein_pos": batch_cpu["protein_pos"][protein_mask],
            "protein_element": batch_cpu["protein_element"][protein_mask],
        }

        pocket_records = []
        pocket_rewards = []
        try:
            temp_dir = TemporaryDirectory(dir=str(reward_dir), prefix=f"online_rl_{pocket_idx:03d}_")
        except Exception:
            total_count += cfg.rollout_num_samples
            invalid_count += cfg.rollout_num_samples
            continue

        with temp_dir as pocket_dir:
            try:
                scorer = VinaPoseScorer(
                    protein=protein,
                    box_ligand_pos=batch_cpu["ligand_pos"][ref_ligand_mask],
                    work_dir=pocket_dir,
                )
            except Exception:
                total_count += cfg.rollout_num_samples
                invalid_count += cfg.rollout_num_samples
                continue
            for sample_idx in range(cfg.rollout_num_samples):
                total_count += 1
                graph_idx = pocket_idx * cfg.rollout_num_samples + sample_idx
                ligand_mask = ligand_batch_all == graph_idx
                if not ligand_mask.any():
                    invalid_count += 1
                    continue

                ligand_pos = sampled_cpu["ligand_pos"][ligand_mask]
                ligand_type = sampled_cpu["ligand_type"][ligand_mask]
                ligand_charge = sampled_cpu["ligand_charge"][ligand_mask]

                global_ligand_idx = torch.where(ligand_mask)[0]
                local_ligand_idx = torch.full((ligand_batch_all.shape[0],), -1, dtype=torch.long)
                local_ligand_idx[global_ligand_idx] = torch.arange(global_ligand_idx.shape[0], dtype=torch.long)
                bond_mask = (
                    ligand_mask[bond_index_all[0]] & ligand_mask[bond_index_all[1]]
                    if bond_index_all.numel() > 0
                    else torch.zeros((0,), dtype=torch.bool)
                )
                ligand_bond_index = local_ligand_idx[bond_index_all[:, bond_mask]]
                ligand_bond_type = bond_type_all[bond_mask]

                try:
                    reward_raw, info = compute_vina_sa_reward(
                        scorer=scorer,
                        ligand_pos=ligand_pos,
                        ligand_type=ligand_type,
                        atom_type_decoder=atom_type_decoder,
                        vina_clip_low=cfg.reward_vina_clip_low,
                        vina_clip_high=cfg.reward_vina_clip_high,
                        vina_offset=cfg.reward_vina_offset,
                        vina_divisor=cfg.reward_vina_divisor,
                        sa_shift=cfg.reward_sa_shift,
                        sa_scale=cfg.reward_sa_scale,
                        ligand_bond_index=ligand_bond_index,
                        ligand_bond_type=ligand_bond_type,
                        ligand_charge=ligand_charge,
                    )
                except Exception:
                    reward_raw, info = None, None

                if reward_raw is None or info is None:
                    invalid_count += 1
                    continue

                pocket_rewards.append(float(reward_raw))
                pocket_records.append(
                    {
                        "protein_pos": batch_cpu["protein_pos"][protein_mask],
                        "protein_element": batch_cpu["protein_element"][protein_mask],
                        "protein_atom_to_aa_type": batch_cpu["protein_atom_to_aa_type"][protein_mask],
                        "protein_is_backbone": batch_cpu["protein_is_backbone"][protein_mask],
                        "ligand_pos": ligand_pos,
                        "ligand_type": ligand_type,
                        "reward_raw": float(reward_raw),
                        "vina_score": float(info["vina_score"]),
                        "sa": float(info["sa"]),
                    }
                )

        if not pocket_records:
            continue

        reward_tensor = torch.tensor(pocket_rewards, dtype=torch.float32)
        mean_c = float(reward_tensor.mean().item())
        std_c = float(reward_tensor.std(unbiased=False).item())
        scale = max(std_c, cfg.reward_min_std)

        for record in pocket_records:
            reward_norm = (float(record["reward_raw"]) - mean_c) / scale
            reward_clip = max(-1.0, min(1.0, reward_norm))
            record["reward"] = 0.5 + 0.5 * reward_clip
            collected.append(record)
            all_raw_rewards.append(float(record["reward_raw"]))

    if all_raw_rewards:
        reward_tensor = torch.tensor(all_raw_rewards, dtype=torch.float32)
        batch_mean = float(reward_tensor.mean().item())
        batch_var = float(reward_tensor.var(unbiased=False).item())
        if reward_ema_mean is None or reward_ema_var is None:
            reward_ema_mean = batch_mean
            reward_ema_var = batch_var
        else:
            reward_ema_mean = (
                cfg.reward_norm_ema_decay * reward_ema_mean
                + (1.0 - cfg.reward_norm_ema_decay) * batch_mean
            )
            reward_ema_var = (
                cfg.reward_norm_ema_decay * reward_ema_var
                + (1.0 - cfg.reward_norm_ema_decay) * batch_var
            )

    reward_std_ema = max(
        (float(reward_ema_var) ** 0.5) if reward_ema_var is not None else 1.0,
        cfg.reward_min_std,
    )
    stats = {
        "num_samples": len(collected),
        "reward_mean": float(torch.tensor(all_raw_rewards).mean().item()) if all_raw_rewards else 0.0,
        "reward_std": float(torch.tensor(all_raw_rewards).std(unbiased=False).item())
        if all_raw_rewards
        else 0.0,
        "reward_std_ema": reward_std_ema,
        "invalid_ratio": invalid_count / max(total_count, 1),
    }
    return collected, stats, reward_ema_mean, reward_ema_var


def train() -> None:
    cfg = load_config(OnlineRLConfig, CONFIG_PATH)
    if cfg.rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be >= 1")
    if cfg.rollout_num_samples <= 0:
        raise ValueError("rollout_num_samples must be >= 1")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype = amp_settings(device, cfg.precision)

    dm = build_datamodule(cfg, batch_size=cfg.rollout_batch_size, drop_last=False)
    if len(dm.ds_train) == 0:
        raise ValueError("online RL training requires a non-empty train split")

    atom_type_decoder = build_atom_type_decoder(dm.ligand_elements, cfg.num_types)
    reward_dir = resolve_path(cfg.reward_work_dir, allow_missing=True)
    reward_dir.mkdir(parents=True, exist_ok=True)

    model = build_diffusion_model(cfg, device)
    anchor_model = build_diffusion_model(cfg, device)
    optimizer = build_optimizer(model, cfg, device)

    base_ckpt = torch.load(resolve_path(cfg.base_ckpt), map_location="cpu")
    base_state = base_ckpt["diffusion"] if "diffusion" in base_ckpt else base_ckpt
    model.load_state_dict(base_state, strict=False)
    anchor_model.load_state_dict(model.state_dict(), strict=True)

    start_update = 0
    best_reward_mean = float("-inf")
    seen_pockets = 0
    reward_ema_mean = None
    reward_ema_var = None
    wandb_id = cfg.run_id

    resume_path = resolve_path(cfg.resume_ckpt)
    if cfg.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["diffusion"], strict=False)
        anchor_model.load_state_dict(ckpt.get("anchor", ckpt["diffusion"]), strict=False)
        optimizer.load_state_dict(ckpt["opt"])
        start_update = int(ckpt.get("update", -1)) + 1
        best_reward_mean = float(ckpt.get("best_reward_mean", float("-inf")))
        seen_pockets = int(ckpt.get("seen_pockets", 0))
        reward_ema_mean = ckpt.get("reward_ema_mean")
        reward_ema_var = ckpt.get("reward_ema_var")
        wandb_id = wandb_id or ckpt.get("wandb_id")

    anchor_model.eval()
    for param in anchor_model.parameters():
        param.requires_grad_(False)

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
    wandb.watch(model, log="gradients", log_freq=50)

    train_iter = iter(dm.train_dataloader())
    for update in tqdm(range(start_update, cfg.updates), desc="online_rl"):
        try:
            batch_cpu = next(train_iter)
        except StopIteration:
            train_iter = iter(dm.train_dataloader())
            batch_cpu = next(train_iter)

        batch_gpu = move_batch_to_device(batch_cpu, device)
        rollouts, rollout_stats, reward_ema_mean, reward_ema_var = collect_rollouts(
            anchor_model=anchor_model,
            batch_cpu=batch_cpu,
            batch_gpu=batch_gpu,
            atom_type_decoder=atom_type_decoder,
            cfg=cfg,
            reward_dir=reward_dir,
            reward_ema_mean=reward_ema_mean,
            reward_ema_var=reward_ema_var,
        )
        seen_pockets += int(batch_cpu["protein_counts"].numel())

        if not rollouts:
            wandb.log(
                {
                    "train/nft_num_samples": 0,
                    "train/nft_reward_mean": rollout_stats["reward_mean"],
                    "train/nft_reward_std": rollout_stats["reward_std"],
                    "train/nft_reward_std_ema": rollout_stats["reward_std_ema"],
                    "train/nft_invalid_ratio": rollout_stats["invalid_ratio"],
                    "seen_pockets": seen_pockets,
                    "update": update,
                },
                step=update,
            )
            continue

        model.train()
        random.shuffle(rollouts)
        total_batches = max(
            1,
            ((len(rollouts) + cfg.batch_size - 1) // cfg.batch_size) * max(1, cfg.num_inner_epochs),
        )
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_pos = 0.0
        total_type = 0.0
        batch_count = 0

        for _ in range(cfg.num_inner_epochs):
            random.shuffle(rollouts)
            for start_idx in range(0, len(rollouts), cfg.batch_size):
                batch_records = rollouts[start_idx : start_idx + cfg.batch_size]
                rl_batch = move_batch_to_device(collate_rollout_batch(batch_records), device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model.nft_loss(
                        rl_batch,
                        anchor_model=anchor_model,
                        reward=rl_batch["reward"],
                        beta=cfg.nft_beta,
                        beta_discrete=cfg.nft_beta_discrete,
                    )
                    loss = out["loss"]

                (loss / total_batches).backward()
                total_loss += float(loss.item())
                total_pos += float(out["loss_position"].item())
                total_type += float(out["loss_atom_type"].item())
                batch_count += 1

        grad_norm = 0.0
        if batch_count > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip).item()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            for anchor_param, model_param in zip(anchor_model.parameters(), model.parameters()):
                anchor_param.mul_(cfg.ema_decay).add_(model_param, alpha=1.0 - cfg.ema_decay)

        denom = max(1, batch_count)
        wandb.log(
            {
                "train/loss": total_loss / denom,
                "train/loss_position": total_pos / denom,
                "train/loss_atom_type": total_type / denom,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": grad_norm,
                "train/nft_num_samples": rollout_stats["num_samples"],
                "train/nft_reward_mean": rollout_stats["reward_mean"],
                "train/nft_reward_std": rollout_stats["reward_std"],
                "train/nft_reward_std_ema": rollout_stats["reward_std_ema"],
                "train/nft_invalid_ratio": rollout_stats["invalid_ratio"],
                "seen_pockets": seen_pockets,
                "update": update,
            },
            step=update,
        )

        payload = {
            "update": update,
            "seen_pockets": seen_pockets,
            "best_reward_mean": best_reward_mean,
            "reward_ema_mean": reward_ema_mean,
            "reward_ema_var": reward_ema_var,
            "wandb_id": wandb_id,
            "diffusion": model.state_dict(),
            "anchor": anchor_model.state_dict(),
            "opt": optimizer.state_dict(),
            "cfg": cfg.__dict__,
        }
        if rollout_stats["reward_mean"] > best_reward_mean:
            best_reward_mean = float(rollout_stats["reward_mean"])
            payload["best_reward_mean"] = best_reward_mean
            torch.save(payload, best_path)
        if (update + 1) % max(1, cfg.log_every) == 0 or update + 1 == cfg.updates:
            torch.save(payload, last_path)

    wandb.finish()


if __name__ == "__main__":
    train()
