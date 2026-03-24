import torch
from tqdm import tqdm

from common import build_datamodule
from config.config import BaseConfig


def update_counts(counts: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().long().view(-1)
    if values.numel() == 0:
        return counts

    max_value = int(values.max().item())
    if max_value >= counts.numel():
        counts = torch.cat(
            [counts, torch.zeros(max_value + 1 - counts.numel(), dtype=torch.long)],
            dim=0,
        )
    counts.scatter_add_(0, values, torch.ones_like(values, dtype=torch.long))
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
    used = (counts > 0).nonzero()
    max_type = int(used.max().item()) if int(used.numel()) > 0 else -1
    print(
        f"{name}: atoms seen={seen:,} | unique types used={int((counts > 0).sum().item())} "
        f"| max type id used={max_type}"
    )
    return counts


def count_atom_types() -> None:
    dm = build_datamodule(BaseConfig(), batch_size=8)
    counts = torch.zeros(0, dtype=torch.long)

    counts = scan_split(dm.train_dataloader(), "train", counts)
    counts = scan_split(dm.val_dataloader(), "val", counts)
    counts = scan_split(dm.test_dataloader(), "test", counts)

    used = (counts > 0).nonzero(as_tuple=False).flatten()
    if used.numel() == 0:
        raise RuntimeError("No ligand types found. Check dataset mapping / collate.")

    print(f"\nRecommended num_types = {int(used.max().item()) + 1}\n")
    print("Top type ids by frequency:")
    for index in torch.topk(counts, k=min(20, counts.numel())).indices.tolist():
        if counts[index].item() > 0:
            print(f"  type={index:>3d}  count={int(counts[index].item()):,}")


if __name__ == "__main__":
    count_atom_types()
