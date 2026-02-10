from pathlib import Path
import torch
from tqdm import tqdm

from model.diffusion import AtomCountPrior, LigandDiffusion
from model.egnn import EGNN
from config.config import InferenceConfig
from datamodules import CrossDockedDataModule


def main():
    cfg = InferenceConfig(ckpt="checkpoints/best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    ligand_diffusion = LigandDiffusion(
        denoiser=denoiser,
        num_types=cfg.num_types,
        steps=cfg.steps,
        type_loss_scale=cfg.type_loss_scale,
        protein_noise_std=cfg.protein_noise_std,
    ).to(device)

    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    ligand_diffusion.load_state_dict(ckpt["diffusion"], strict=True)
    ligand_diffusion.eval()

    prior = AtomCountPrior.from_state_dict(ckpt["prior"]) if isinstance(ckpt, dict) and "prior" in ckpt else AtomCountPrior.fit(dm.ds_train, n_bins=10)

    out_dir = Path("inference") / Path(cfg.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = ["protein_pos", "protein_batch", "protein_element", "protein_atom_to_aa_type", "protein_is_backbone"]
    amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp else torch.float32

    for i, batch in enumerate(tqdm(dm.test_dataloader(), desc="sample")):
        protein_batch_dict = {k: batch[k].to(device, non_blocking=True) for k in keys if k in batch}

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            sample = ligand_diffusion.sample(protein_batch_dict, prior)

        out = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sample.items()}
        torch.save(out, out_dir / f"sample_{i:06d}.pt") # save each predicted complex to a separate file


if __name__ == "__main__":
    main()
