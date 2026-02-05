from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import lmdb
import lightning.pytorch as pl


def map_elements(z: torch.Tensor, element_map: Dict[int, int], unknown_index: int) -> torch.Tensor:
    z_cpu = z.detach().to("cpu")
    out = torch.empty_like(z_cpu, dtype=torch.long)
    for i in range(z_cpu.numel()):
        out[i] = element_map.get(int(z_cpu[i].item()), unknown_index)
    return out.to(device=z.device)


@dataclass
class CrossDockedExample:
    protein_pos: torch.Tensor
    protein_element: torch.Tensor
    protein_is_backbone: torch.Tensor
    protein_atom_to_aa_type: torch.Tensor

    ligand_pos: torch.Tensor
    ligand_element: torch.Tensor
    ligand_bond_index: torch.Tensor
    ligand_bond_type: torch.Tensor

    ligand_atom_feature: Optional[torch.Tensor] = None
    ligand_center_of_mass: Optional[torch.Tensor] = None

    protein_molecule_name: Optional[str] = None
    ligand_smiles: Optional[str] = None
    protein_filename: Optional[str] = None
    ligand_filename: Optional[str] = None


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
    ):
        self.lmdb_path = lmdb_path
        self.split_pt_path = split_pt_path
        self.split = split
        self.center = center
        self.return_text_fields = return_text_fields

        split_obj = torch.load(split_pt_path, map_location="cpu")
        if not isinstance(split_obj, dict) or split not in split_obj:
            raise ValueError(f"split file must be dict with key '{split}', got keys={list(split_obj.keys()) if isinstance(split_obj, dict) else type(split_obj)}")

        indices = split_obj[split]
        if not isinstance(indices, list) or (len(indices) > 0 and not isinstance(indices[0], int)):
            raise ValueError(f"expected split['{split}'] to be a list[int], got {type(indices)}")

        self.keys = [str(i).encode() for i in indices]

        ligand_elements_list = list(ligand_elements)
        if include_unknown_ligand_type:
            self.ligand_unknown_index = len(ligand_elements_list)
        else:
            self.ligand_unknown_index = -1

        self.ligand_element_map = {int(z): i for i, z in enumerate(ligand_elements_list)}

        self.env = None

    def open_env(self) -> lmdb.Environment:
        if self.env is not None:
            return self.env
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

    def __getitem__(self, i: int) -> Dict[str, Any]:
        worker = get_worker_info()
        if worker is not None and self.env is None:
            self.open_env()

        env = self.open_env()
        key = self.keys[i]
        with env.begin(write=False) as txn:
            buf = txn.get(key)
        if buf is None:
            raise KeyError(f"missing lmdb key {key!r}")

        ex = pickle.loads(buf)
        if not isinstance(ex, dict):
            raise TypeError(f"lmdb record must be dict, got {type(ex)}")

        protein_pos = ex["protein_pos"].float()
        ligand_pos = ex["ligand_pos"].float()

        if self.center == "protein_mean":
            shift = protein_pos.mean(dim=0)
            protein_pos = protein_pos - shift
            ligand_pos = ligand_pos - shift
        elif self.center == "none":
            pass
        else:
            raise ValueError("center must be 'protein_mean' or 'none'")

        protein_element = ex["protein_element"].long()
        protein_is_backbone = ex["protein_is_backbone"].bool()
        protein_atom_to_aa_type = ex["protein_atom_to_aa_type"].long()

        ligand_element = ex["ligand_element"].long()
        ligand_bond_index = ex["ligand_bond_index"].long()
        ligand_bond_type = ex["ligand_bond_type"].long()

        if self.ligand_unknown_index >= 0:
            ligand_type = map_elements(ligand_element, self.ligand_element_map, self.ligand_unknown_index)
        else:
            ligand_type = map_elements(ligand_element, self.ligand_element_map, 0)

        out: Dict[str, Any] = {
            "key": key,
            "protein_pos": protein_pos,
            "protein_element": protein_element,
            "protein_is_backbone": protein_is_backbone,
            "protein_atom_to_aa_type": protein_atom_to_aa_type,
            "ligand_pos": ligand_pos,
            "ligand_element": ligand_element,
            "ligand_type": ligand_type,
            "ligand_bond_index": ligand_bond_index,
            "ligand_bond_type": ligand_bond_type,
        }

        if "ligand_atom_feature" in ex:
            out["ligand_atom_feature"] = ex["ligand_atom_feature"].long()
        if "ligand_center_of_mass" in ex:
            out["ligand_center_of_mass"] = ex["ligand_center_of_mass"].float()

        if self.return_text_fields:
            out["protein_molecule_name"] = ex.get("protein_molecule_name", None)
            out["ligand_smiles"] = ex.get("ligand_smiles", None)
            out["protein_filename"] = ex.get("protein_filename", None)
            out["ligand_filename"] = ex.get("ligand_filename", None)

        return out


def collate_crossdocked(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    protein_pos_list = []
    protein_element_list = []
    protein_is_backbone_list = []
    protein_aa_type_list = []
    protein_batch_list = []

    ligand_pos_list = []
    ligand_element_list = []
    ligand_type_list = []
    ligand_batch_list = []

    bond_index_list = []
    bond_type_list = []
    bond_batch_list = []

    ligand_atom_feature_list = []
    has_ligand_atom_feature = all(("ligand_atom_feature" in ex) for ex in batch)

    keys = []
    protein_counts = []
    ligand_counts = []

    p_offset = 0
    l_offset = 0

    for b, ex in enumerate(batch):
        keys.append(ex["key"])

        ppos = ex["protein_pos"]
        protein_counts.append(int(ppos.shape[0]))
        protein_pos_list.append(ppos)
        protein_element_list.append(ex["protein_element"])
        protein_is_backbone_list.append(ex["protein_is_backbone"])
        protein_aa_type_list.append(ex["protein_atom_to_aa_type"])
        protein_batch_list.append(torch.full((ppos.shape[0],), b, dtype=torch.long))

        lpos = ex["ligand_pos"]
        ligand_counts.append(int(lpos.shape[0]))
        ligand_pos_list.append(lpos)
        ligand_element_list.append(ex["ligand_element"])
        ligand_type_list.append(ex["ligand_type"])
        ligand_batch_list.append(torch.full((lpos.shape[0],), b, dtype=torch.long))

        bidx = ex["ligand_bond_index"]
        btyp = ex["ligand_bond_type"]
        if bidx.numel() > 0:
            bond_index_list.append(bidx + l_offset)
            bond_type_list.append(btyp)
            bond_batch_list.append(torch.full((btyp.shape[0],), b, dtype=torch.long))

        if has_ligand_atom_feature:
            ligand_atom_feature_list.append(ex["ligand_atom_feature"].long())

        p_offset += int(ppos.shape[0])
        l_offset += int(lpos.shape[0])

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
        "ligand_batch": torch.cat(ligand_batch_list, dim=0),
        "ligand_counts": torch.tensor(ligand_counts, dtype=torch.long),
    }

    if len(bond_index_list) > 0:
        out["ligand_bond_index"] = torch.cat(bond_index_list, dim=1)
        out["ligand_bond_type"] = torch.cat(bond_type_list, dim=0)
        out["ligand_bond_batch"] = torch.cat(bond_batch_list, dim=0)
    else:
        out["ligand_bond_index"] = torch.empty((2, 0), dtype=torch.long)
        out["ligand_bond_type"] = torch.empty((0,), dtype=torch.long)
        out["ligand_bond_batch"] = torch.empty((0,), dtype=torch.long)

    if has_ligand_atom_feature:
        out["ligand_atom_feature"] = torch.cat(ligand_atom_feature_list, dim=0)

    return out


class CrossDockedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        lmdb_path: str,
        split_pt_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        center: str = "protein_mean",
        ligand_elements: Sequence[int] = (6, 7, 8, 9, 15, 16, 17, 35, 53),
        include_unknown_ligand_type: bool = True,
        return_text_fields: bool = False,
        drop_last: bool = True,
    ):
        self.lmdb_path = lmdb_path
        self.split_pt_path = split_pt_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.center = center
        self.ligand_elements = tuple(int(x) for x in ligand_elements)
        self.include_unknown_ligand_type = bool(include_unknown_ligand_type)
        self.return_text_fields = bool(return_text_fields)
        self.drop_last = bool(drop_last)

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = CrossDockedLMDBDataset(
            lmdb_path=self.lmdb_path,
            split_pt_path=self.split_pt_path,
            split="train",
            center=self.center,
            ligand_elements=self.ligand_elements,
            include_unknown_ligand_type=self.include_unknown_ligand_type,
            return_text_fields=self.return_text_fields,
        )
        self.ds_val = CrossDockedLMDBDataset(
            lmdb_path=self.lmdb_path,
            split_pt_path=self.split_pt_path,
            split="val",
            center=self.center,
            ligand_elements=self.ligand_elements,
            include_unknown_ligand_type=self.include_unknown_ligand_type,
            return_text_fields=self.return_text_fields,
        )
        self.ds_test = CrossDockedLMDBDataset(
            lmdb_path=self.lmdb_path,
            split_pt_path=self.split_pt_path,
            split="test",
            center=self.center,
            ligand_elements=self.ligand_elements,
            include_unknown_ligand_type=self.include_unknown_ligand_type,
            return_text_fields=self.return_text_fields,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            drop_last=self.drop_last,
            collate_fn=collate_crossdocked,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            drop_last=False,
            collate_fn=collate_crossdocked,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            drop_last=False,
            collate_fn=collate_crossdocked,
        )
