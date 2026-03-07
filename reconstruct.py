# use openbabel to build molecules from the predicted geometry and atom types

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from openbabel import openbabel
from rdkit import Chem


DEFAULT_LIGAND_ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53)


@dataclass
class BuildResult:
    sample_idx: int
    ok: bool
    mol: Any | None
    error: str | None = None


def list_target_files(inference_dir: Path) -> list[Path]:
    return sorted(
        path / "target.pt"
        for path in inference_dir.iterdir()
        if path.is_dir() and (path / "target.pt").exists()
    )


def load_target_file(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    needed = {"target_idx", "protein", "reference", "samples"}
    missing = needed.difference(data.keys())
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return data


def infer_mol_from_geometry(
    ligand_pos: torch.Tensor,
    ligand_type: torch.Tensor,
    atom_type_decoder: dict[int, Any],
) -> Any | None:
    if ligand_pos.numel() == 0 or ligand_type.numel() == 0:
        return None
    if ligand_pos.ndim != 2 or ligand_pos.shape[1] != 3:
        raise ValueError(f"ligand_pos must have shape [N, 3], got {tuple(ligand_pos.shape)}")
    if ligand_type.ndim != 1 or ligand_type.shape[0] != ligand_pos.shape[0]:
        raise ValueError(
            f"ligand_type must have shape [N] matching ligand_pos, got {tuple(ligand_type.shape)}"
        )

    pt = Chem.GetPeriodicTable()
    decoder = atom_type_decoder or {
        i: atomic_num for i, atomic_num in enumerate(DEFAULT_LIGAND_ELEMENTS)
    }

    def decode_atomic_num(type_id: int) -> int:
        if type_id not in decoder:
            valid = sorted(decoder.keys())
            raise ValueError(f"unknown ligand_type index {type_id}; valid indices are {valid}")
        value = decoder[type_id]
        if isinstance(value, dict):
            atomic_num = value.get("atomic_num", value.get("atomic_number", value.get("element")))
        else:
            atomic_num = value

        if isinstance(atomic_num, str):
            atomic_num = pt.GetAtomicNumber(atomic_num)
        atomic_num = int(atomic_num)
        if atomic_num <= 0:
            raise ValueError(f"invalid atomic number decoded from type {type_id}: {atomic_num}")
        return atomic_num

    coords = ligand_pos.detach().cpu().to(torch.float64)
    type_ids = ligand_type.detach().cpu().to(torch.long).tolist()
    if min(type_ids) < 0:
        raise ValueError(f"ligand_type contains negative indices: min={min(type_ids)}")

    ob_mol = openbabel.OBMol()
    ob_mol.BeginModify()
    for xyz, t in zip(coords.tolist(), type_ids):
        atom = ob_mol.NewAtom()
        atom.SetAtomicNum(decode_atomic_num(int(t)))
        atom.SetVector(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ob_mol.EndModify()

    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()

    conv = openbabel.OBConversion()
    conv.SetOutFormat("mol")
    mol_block = conv.WriteString(ob_mol)
    rd_mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if rd_mol is None:
        return None

    try:
        Chem.SanitizeMol(rd_mol)
    except Exception:
        try:
            Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception:
            return None
    return rd_mol


def build_sample(
    sample: dict[str, Any],
    atom_type_decoder: dict[int, Any],
) -> BuildResult:
    sample_idx = int(sample.get("sample_idx", -1))

    try:
        ligand_pos = sample["ligand_pos"]
        ligand_type = sample["ligand_type"]

        mol = infer_mol_from_geometry(
            ligand_pos=ligand_pos,
            ligand_type=ligand_type,
            atom_type_decoder=atom_type_decoder,
        )

        if mol is None:
            return BuildResult(
                sample_idx=sample_idx,
                ok=False,
                mol=None,
                error="invalid_molecule",
            )

        return BuildResult(
            sample_idx=sample_idx,
            ok=True,
            mol=mol,
            error=None,
        )
    except Exception as exc:
        return BuildResult(
            sample_idx=sample_idx,
            ok=False,
            mol=None,
            error=str(exc),
        )


def build_target(
    target: dict[str, Any],
    atom_type_decoder: dict[int, Any],
) -> dict[str, int | list[BuildResult] | dict[str, Any]]:
    built_samples = [
        build_sample(sample, atom_type_decoder)
        for sample in target["samples"]
    ]

    return {
        "target_idx": int(target["target_idx"]),
        "protein": target["protein"],
        "reference": target["reference"],
        "samples": target["samples"],
        "built_samples": built_samples,
    }


def valid_results(results: list[BuildResult]) -> list[BuildResult]:
    return [result for result in results if result.ok]


def valid_fraction(results: list[BuildResult]) -> float:
    if not results:
        return 0.0
    return len(valid_results(results)) / len(results)
