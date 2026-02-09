import torch
from tqdm import tqdm
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamodules import CrossDockedDataModule

lmdb_path = os.path.abspath("data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")
split_path = os.path.abspath("data/crossdocked_pose_split_from_name_val1000.pt")


def update_counts(counts: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to("cpu").long().view(-1)
    if x.numel() == 0:
        return counts
    mx = int(x.max().item())
    if mx >= counts.numel():
        counts = torch.cat([counts, torch.zeros(mx + 1 - counts.numel(), dtype=torch.long)], dim=0)
    counts.scatter_add_(0, x, torch.ones_like(x, dtype=torch.long))
    return counts


@torch.no_grad()
def scan_split(loader, name: str, counts: torch.Tensor) -> torch.Tensor:
    print(f"Scanning {name} split for ligand type counts...")
    seen = 0
    for batch in tqdm(loader):
        if "ligand_type" not in batch:
            raise KeyError("batch missing 'ligand_type' (this should be mapped element categories)")
        counts = update_counts(counts, batch["ligand_type"])
        seen += int(batch["ligand_type"].numel())
    total = int(counts.sum().item())
    uniq = int((counts > 0).sum().item())
    mx = int((counts > 0).nonzero().max().item()) if uniq > 0 else -1
    print(f"{name}: atoms seen={seen:,} | unique types used={uniq} | max type id used={mx}")
    return counts


def count_atom_types():
    dm = CrossDockedDataModule(
        lmdb_path=lmdb_path,
        split_pt_path=split_path,
        batch_size=8,
        num_workers=2,
    )
    dm.setup()

    counts = torch.zeros(0, dtype=torch.long)

    counts = scan_split(dm.train_dataloader(), "train", counts)
    counts = scan_split(dm.val_dataloader(), "val", counts)
    counts = scan_split(dm.test_dataloader(), "test", counts)

    used = (counts > 0).nonzero(as_tuple=False).flatten()
    if used.numel() == 0:
        raise RuntimeError("No ligand types found. Check dataset mapping / collate.")

    num_types = int(used.max().item()) + 1
    print()
    print(f"Recommended num_types = {num_types}")
    print()

    top = torch.topk(counts, k=min(20, counts.numel())).indices.tolist()
    print("Top type ids by frequency:")
    for i in top:
        if counts[i].item() == 0:
            continue
        print(f"  type={i:>3d}  count={int(counts[i].item()):,}")

if __name__ == "__main__":
    count_atom_types()