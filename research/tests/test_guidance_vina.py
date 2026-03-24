import json
import math
import pickle
from pathlib import Path
import pytest
import lmdb
import torch

from config.config import GuidanceConfig
from datamodules import CrossDockedDataModule
from common import DEFAULT_GUIDANCE_TARGET, vina_score_metadata
from model.egnn import EGNN
from model.sampling_guidance import PocketAffinityGuidance
from scripts.train_guidance import eval_epoch, train_epoch
from common import score_bound_pose
from reconstruct_molecule import build_reference_mol

def _toy_record(shift: float = 0.0) -> dict:
    return {
        "protein_pos": torch.tensor(
            [
                [0.0 + shift, 0.0, 0.0],
                [4.0 + shift, 0.0, 0.0],
                [0.0 + shift, 4.0, 0.0],
                [0.0 + shift, 0.0, 4.0],
                [3.0 + shift, 3.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        "protein_element": torch.tensor([6, 7, 8, 16, 6], dtype=torch.long),
        "protein_is_backbone": torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool),
        "protein_atom_to_aa_type": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "ligand_pos": torch.tensor(
            [
                [1.2 + shift, 1.0, 1.0],
                [2.5 + shift, 1.0, 1.0],
                [3.4 + shift, 1.5, 1.0],
            ],
            dtype=torch.float32,
        ),
        "ligand_element": torch.tensor([6, 6, 8], dtype=torch.long),
        "ligand_bond_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        "ligand_bond_type": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        "ligand_smiles": "CCO",
    }


def _write_toy_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    lmdb_path = tmp_path / "toy.lmdb"
    env = lmdb.open(str(lmdb_path), subdir=False, map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(b"0", pickle.dumps(_toy_record(shift=0.0)))
        txn.put(b"1", pickle.dumps(_toy_record(shift=0.2)))
    env.close()

    split_path = tmp_path / "split.pt"
    torch.save({"train": [0], "val": [1], "test": []}, split_path)

    label_path = tmp_path / "vina_labels.jsonl"
    with label_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(vina_score_metadata()) + "\n")
        f.write(
            json.dumps(
                {
                    "kind": "sample",
                    "sample_key": "0",
                    "split": "train",
                    "split_index": 0,
                    "status": "ok",
                    DEFAULT_GUIDANCE_TARGET: -7.25,
                    "error": None,
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "kind": "sample",
                    "sample_key": "1",
                    "split": "val",
                    "split_index": 0,
                    "status": "ok",
                    DEFAULT_GUIDANCE_TARGET: -6.80,
                    "error": None,
                }
            )
            + "\n"
        )

    return lmdb_path, split_path, label_path


def test_vina_score_bound_pose_returns_finite_value(tmp_path: Path) -> None:

    sample = _toy_record()
    mol = build_reference_mol(sample)
    protein = {
        "protein_pos": sample["protein_pos"],
        "protein_element": sample["protein_element"],
    }

    score = score_bound_pose(
        protein=protein,
        mol=mol,
        ligand_pos=sample["ligand_pos"],
        work_dir=tmp_path,
    )

    assert math.isfinite(score)


def test_datamodule_loads_vina_sidecar_labels(tmp_path: Path) -> None:
    lmdb_path, split_path, label_path = _write_toy_dataset(tmp_path)

    dm = CrossDockedDataModule(
        lmdb_path=str(lmdb_path),
        split_pt_path=str(split_path),
        batch_size=1,
        num_workers=0,
        guidance_label_path=str(label_path),
        guidance_target=DEFAULT_GUIDANCE_TARGET,
        drop_last=False,
    )

    batch = next(iter(dm.train_dataloader()))

    assert DEFAULT_GUIDANCE_TARGET in batch
    assert batch[DEFAULT_GUIDANCE_TARGET].shape == (1,)
    assert float(batch[DEFAULT_GUIDANCE_TARGET][0].item()) == pytest.approx(-7.25)


def test_guidance_train_epoch_runs_with_vina_score_labels(tmp_path: Path) -> None:
    lmdb_path, split_path, label_path = _write_toy_dataset(tmp_path)

    cfg = GuidanceConfig(
        lmdb_path=str(lmdb_path),
        split_path=str(split_path),
        batch_size=1,
        num_workers=0,
        epochs=1,
        hidden_dim=32,
        num_layers=1,
        num_r_gaussian=8,
        k=4,
        message_passing_mode="mlp",
        guidance_target=DEFAULT_GUIDANCE_TARGET,
        guidance_label_path=str(label_path),
        precision="fp32",
    )

    dm = CrossDockedDataModule(
        lmdb_path=cfg.lmdb_path,
        split_pt_path=cfg.split_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        guidance_label_path=cfg.guidance_label_path,
        guidance_target=cfg.guidance_target,
        drop_last=False,
    )

    device = torch.device("cpu")
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
    model = PocketAffinityGuidance(
        denoiser=denoiser,
        num_types=cfg.num_types,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_stats = train_epoch(
        model,
        dm.train_dataloader(),
        optimizer,
        device,
        target_name=cfg.guidance_target,
        target_loss_scale=cfg.target_loss_scale,
        use_amp=False,
        amp_dtype=torch.float32,
        grad_clip=cfg.grad_clip,
    )
    val_stats = eval_epoch(
        model,
        dm.val_dataloader(),
        device,
        target_name=cfg.guidance_target,
    )

    assert train_stats["n"] == 1
    assert val_stats["n"] == 1
    assert math.isfinite(train_stats["loss"])
    assert math.isfinite(val_stats["loss"])
