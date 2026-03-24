from typing import Any, Dict, Sequence
import pickle

import lmdb
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from common import load_scalar_labels
from reconstruct_molecule import formal_charges_from_reference


DATASET_SPLITS = ("train", "val", "test")
SCALAR_SKIP_KEYS = {
    "key",
    "protein_pos",
    "protein_element",
    "protein_is_backbone",
    "protein_atom_to_aa_type",
    "ligand_pos",
    "ligand_element",
    "ligand_type",
    "ligand_bond_index",
    "ligand_bond_type",
    "ligand_formal_charge",
    "ligand_atom_feature",
    "ligand_center_of_mass",
    "ligand_smiles",
}


def map_ligand_elements(
    atomic_numbers: torch.Tensor,
    element_map: dict[int, int],
    unknown_index: int,
) -> torch.Tensor:
    mapped = [element_map.get(int(value), unknown_index) for value in atomic_numbers.detach().cpu().tolist()]
    return torch.tensor(mapped, dtype=torch.long, device=atomic_numbers.device)


class CrossDockedLMDBDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        split_pt_path: str,
        split: str,
        center: str = "protein_mean",
        ligand_elements: Sequence[int] = (6, 7, 8, 9, 15, 16, 17, 35, 53),
        include_unknown_ligand_type: bool = True,
        return_text_fields: bool = False,
        guidance_label_path: str | None = None,
        guidance_target: str | None = None,
    ):
        self.lmdb_path = lmdb_path
        self.split_pt_path = split_pt_path
        self.split = split
        self.center = center
        self.return_text_fields = return_text_fields
        self.guidance_target = guidance_target

        split_obj = torch.load(split_pt_path, map_location="cpu")
        if not isinstance(split_obj, dict) or split not in split_obj:
            keys = list(split_obj.keys()) if isinstance(split_obj, dict) else type(split_obj)
            raise ValueError(f"split file must be dict with key {split!r}, got {keys}")

        indices = split_obj[split]
        if not isinstance(indices, list) or any(not isinstance(index, int) for index in indices):
            raise ValueError(f"expected split[{split!r}] to be list[int], got {type(indices)}")

        self.scalar_labels: dict[str, float] = {}
        self.keys = [str(index).encode() for index in indices]
        if guidance_label_path is not None:
            if guidance_target is None:
                raise ValueError("guidance_target must be provided when guidance_label_path is set")
            self.scalar_labels = load_scalar_labels(
                guidance_label_path,
                target_name=guidance_target,
                split=split,
            )
            self.keys = [key for key in self.keys if key.decode("utf-8") in self.scalar_labels]

        self.ligand_elements = tuple(int(value) for value in ligand_elements)
        self.ligand_unknown_index = (
            len(self.ligand_elements) if include_unknown_ligand_type else -1
        )
        self.ligand_element_map = {
            atomic_num: index for index, atomic_num in enumerate(self.ligand_elements)
        }
        self.env: lmdb.Environment | None = None

    def open_env(self) -> lmdb.Environment:
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=256,
                meminit=False,
            )
        return self.env

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if get_worker_info() is not None and self.env is None:
            self.open_env()

        with self.open_env().begin(write=False) as txn:
            buf = txn.get(self.keys[index])
        if buf is None:
            raise KeyError(f"missing lmdb key {self.keys[index]!r}")

        record = pickle.loads(buf)
        if not isinstance(record, dict):
            raise TypeError(f"lmdb record must be dict, got {type(record)}")

        protein_pos = record["protein_pos"].float()
        ligand_pos = record["ligand_pos"].float()
        if self.center == "protein_mean":
            shift = protein_pos.mean(dim=0)
            protein_pos = protein_pos - shift
            ligand_pos = ligand_pos - shift
        elif self.center != "none":
            raise ValueError("center must be 'protein_mean' or 'none'")

        ligand_element = record["ligand_element"].long()
        unknown_index = self.ligand_unknown_index if self.ligand_unknown_index >= 0 else 0

        batch: Dict[str, Any] = {
            "key": self.keys[index],
            "protein_pos": protein_pos,
            "protein_element": record["protein_element"].long(),
            "protein_is_backbone": record["protein_is_backbone"].bool(),
            "protein_atom_to_aa_type": record["protein_atom_to_aa_type"].long(),
            "ligand_pos": ligand_pos,
            "ligand_element": ligand_element,
            "ligand_type": map_ligand_elements(
                ligand_element,
                self.ligand_element_map,
                unknown_index,
            ),
            "ligand_bond_index": record["ligand_bond_index"].long(),
            "ligand_bond_type": record["ligand_bond_type"].long(),
            "ligand_formal_charge": formal_charges_from_reference(record),
        }

        if "ligand_atom_feature" in record:
            batch["ligand_atom_feature"] = record["ligand_atom_feature"].long()
        if "ligand_center_of_mass" in record:
            batch["ligand_center_of_mass"] = record["ligand_center_of_mass"].float()
        if "affinity" in record:
            batch["affinity"] = torch.tensor(float(record["affinity"]), dtype=torch.float32)
        if self.guidance_target is not None:
            key_str = self.keys[index].decode("utf-8")
            if key_str in self.scalar_labels:
                batch[self.guidance_target] = torch.tensor(
                    float(self.scalar_labels[key_str]),
                    dtype=torch.float32,
                )
        if self.return_text_fields:
            batch["ligand_smiles"] = record.get("ligand_smiles")

        return batch


def collate_crossdocked(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    protein_pos_list = []
    protein_element_list = []
    protein_is_backbone_list = []
    protein_aa_type_list = []
    protein_batch_list = []

    ligand_pos_list = []
    ligand_element_list = []
    ligand_type_list = []
    ligand_charge_list = []
    ligand_batch_list = []

    bond_index_list = []
    bond_type_list = []
    bond_batch_list = []

    ligand_atom_feature_list = []
    keys = []
    protein_counts = []
    ligand_counts = []

    has_ligand_atom_feature = all("ligand_atom_feature" in row for row in batch)
    has_ligand_smiles = all("ligand_smiles" in row for row in batch)

    ligand_offset = 0
    for batch_index, row in enumerate(batch):
        keys.append(row["key"])

        protein_pos = row["protein_pos"]
        ligand_pos = row["ligand_pos"]
        protein_counts.append(int(protein_pos.shape[0]))
        ligand_counts.append(int(ligand_pos.shape[0]))

        protein_pos_list.append(protein_pos)
        protein_element_list.append(row["protein_element"])
        protein_is_backbone_list.append(row["protein_is_backbone"])
        protein_aa_type_list.append(row["protein_atom_to_aa_type"])
        protein_batch_list.append(torch.full((protein_pos.shape[0],), batch_index, dtype=torch.long))

        ligand_pos_list.append(ligand_pos)
        ligand_element_list.append(row["ligand_element"])
        ligand_type_list.append(row["ligand_type"])
        ligand_charge_list.append(row["ligand_formal_charge"])
        ligand_batch_list.append(torch.full((ligand_pos.shape[0],), batch_index, dtype=torch.long))

        bond_index = row["ligand_bond_index"]
        bond_type = row["ligand_bond_type"]
        if bond_index.numel() > 0:
            bond_index_list.append(bond_index + ligand_offset)
            bond_type_list.append(bond_type)
            bond_batch_list.append(torch.full((bond_type.shape[0],), batch_index, dtype=torch.long))

        if has_ligand_atom_feature:
            ligand_atom_feature_list.append(row["ligand_atom_feature"].long())

        ligand_offset += int(ligand_pos.shape[0])

    out: Dict[str, Any] = {
        "keys": keys,
        "protein_pos": torch.cat(protein_pos_list, dim=0),
        "protein_element": torch.cat(protein_element_list, dim=0),
        "protein_is_backbone": torch.cat(protein_is_backbone_list, dim=0),
        "protein_atom_to_aa_type": torch.cat(protein_aa_type_list, dim=0),
        "protein_batch": torch.cat(protein_batch_list, dim=0),
        "protein_counts": torch.tensor(protein_counts, dtype=torch.long),
        "ligand_pos": torch.cat(ligand_pos_list, dim=0),
        "ligand_element": torch.cat(ligand_element_list, dim=0),
        "ligand_type": torch.cat(ligand_type_list, dim=0),
        "ligand_formal_charge": torch.cat(ligand_charge_list, dim=0),
        "ligand_batch": torch.cat(ligand_batch_list, dim=0),
        "ligand_counts": torch.tensor(ligand_counts, dtype=torch.long),
    }
    if bond_index_list:
        out["ligand_bond_index"] = torch.cat(bond_index_list, dim=1)
        out["ligand_bond_type"] = torch.cat(bond_type_list, dim=0)
        out["ligand_bond_batch"] = torch.cat(bond_batch_list, dim=0)
    else:
        out["ligand_bond_index"] = torch.empty((2, 0), dtype=torch.long)
        out["ligand_bond_type"] = torch.empty((0,), dtype=torch.long)
        out["ligand_bond_batch"] = torch.empty((0,), dtype=torch.long)

    if has_ligand_atom_feature:
        out["ligand_atom_feature"] = torch.cat(ligand_atom_feature_list, dim=0)
    if has_ligand_smiles:
        out["ligand_smiles"] = [row["ligand_smiles"] for row in batch]

    for key, value in batch[0].items():
        if key in SCALAR_SKIP_KEYS or not torch.is_tensor(value) or value.ndim != 0:
            continue
        if all(torch.is_tensor(row.get(key)) and row[key].ndim == 0 for row in batch):
            out[key] = torch.stack([row[key] for row in batch], dim=0)

    return out


class CrossDockedDataModule:
    def __init__(
        self,
        lmdb_path: str = "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
        split_pt_path: str = "data/crossdocked_pose_split_from_name_val1000.pt",
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        center: str = "protein_mean",
        ligand_elements: Sequence[int] = (6, 7, 8, 9, 15, 16, 17, 35, 53),
        include_unknown_ligand_type: bool = True,
        return_text_fields: bool = False,
        guidance_label_path: str | None = None,
        guidance_target: str | None = None,
        drop_last: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.drop_last = bool(drop_last)
        self.ligand_elements = tuple(int(value) for value in ligand_elements)

        dataset_kwargs = {
            "lmdb_path": lmdb_path,
            "split_pt_path": split_pt_path,
            "center": center,
            "ligand_elements": self.ligand_elements,
            "include_unknown_ligand_type": include_unknown_ligand_type,
            "return_text_fields": return_text_fields,
            "guidance_label_path": guidance_label_path,
            "guidance_target": guidance_target,
        }
        self.datasets = {
            split: CrossDockedLMDBDataset(split=split, **dataset_kwargs)
            for split in DATASET_SPLITS
        }
        self.ds_train = self.datasets["train"]
        self.ds_val = self.datasets["val"]
        self.ds_test = self.datasets["test"]

    def dataloader(self, split: str, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=drop_last,
            collate_fn=collate_crossdocked,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader("train", shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader("val", shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader("test", shuffle=False, drop_last=False)



def build_datamodule(cfg: Any, **overrides: Any) -> CrossDockedDataModule:

    params: dict[str, Any] = {
        "lmdb_path": str(overrides.pop("lmdb_path", cfg.lmdb_path)),
        "split_pt_path": overrides.pop("split_pt_path", getattr(cfg, "split_path")),
        "batch_size": overrides.pop("batch_size", cfg.batch_size),
        "num_workers": overrides.pop("num_workers", cfg.num_workers),
        "pin_memory": overrides.pop("pin_memory", cfg.pin_memory),
        "persistent_workers": overrides.pop("persistent_workers", cfg.persistent_workers),
        "prefetch_factor": overrides.pop("prefetch_factor", cfg.prefetch_factor),
    }
    for key in (
        "center",
        "ligand_elements",
        "include_unknown_ligand_type",
        "return_text_fields",
        "guidance_target",
        "drop_last",
    ):
        value = overrides.pop(key, getattr(cfg, key, None))
        if value is not None:
            params[key] = value

    guidance_label_path = overrides.pop("guidance_label_path", getattr(cfg, "guidance_label_path", None))
    if guidance_label_path is not None:
        params["guidance_label_path"] = guidance_label_path

    params.update(overrides)
    return CrossDockedDataModule(**params)