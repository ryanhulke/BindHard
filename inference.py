from pathlib import Path
import torch
from tqdm import tqdm
import yaml

from model.diffusion import AtomCountPrior, LigandDiffusion
from model.egnn import EGNN
from config.config import InferenceConfig
from datamodules import CrossDockedDataModule


def main():
    with open("config/inference/base_config.yaml", "r") as f:
        cfg = InferenceConfig(**yaml.safe_load(f))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_per_target = getattr(cfg, "samples_per_target", 100)
    save_trajectory = getattr(cfg, "save_trajectory", True)

    dm = CrossDockedDataModule(
        lmdb_path=cfg.lmdb_path,
        split_pt_path=cfg.split_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    denoiser = EGNN(
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        edge_feat_dim=cfg.edge_feat_dim,
        num_r_gaussian=cfg.num_r_gaussian,
        message_passing_mode=cfg.message_passing_mode,
        k=cfg.k,
        cutoff_mode=cfg.cutoff_mode,
        update_x=True
    ).to(device)

    model = LigandDiffusion(
        denoiser=denoiser,
        num_types=cfg.num_types,
        steps=cfg.steps,
        type_loss_scale=cfg.type_loss_scale,
        protein_noise_std=cfg.protein_noise_std,
    ).to(device)

    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["diffusion"], strict=True)
    model.eval()

    if "prior" in ckpt:
        prior = AtomCountPrior.from_state_dict(ckpt["prior"])
    else:
        print("fitting atom count prior...")
        prior = AtomCountPrior.fit(dm.ds_train, n_bins=10)

    out_dir = Path("inference") / Path(cfg.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_keys = [
        "protein_pos",
        "protein_batch",
        "protein_element",
        "protein_atom_to_aa_type",
        "protein_is_backbone",
    ]

    ref_keys = [
        "ligand_pos",
        "ligand_element",
        "ligand_bond_index",
        "ligand_bond_type",
        "affinity",
    ]

    amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp else torch.float32
    for target_idx, batch in enumerate(tqdm(dm.test_dataloader(), desc="target")):
        protein = {k: batch[k].to(device, non_blocking=True) for k in protein_keys}
        samples = []

        for sample_idx in tqdm(range(samples_per_target)):
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp,
            ):
                if save_trajectory:
                    sample = model.sample(protein, prior, return_trajectory=True, trajectory_stride=10)
                else:
                    sample = model.sample(protein, prior)

            sample = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in sample.items()
            }
            sample["sample_idx"] = sample_idx
            samples.append(sample)

        target_dir = out_dir / f"{target_idx:06d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "target_idx": target_idx,
                "protein": {k: batch[k].cpu() for k in protein_keys},
                "reference": {k: batch[k].cpu() for k in ref_keys if k in batch},
                "samples": samples,
            },
            target_dir / "target.pt",
        )


if __name__ == "__main__":
    main()