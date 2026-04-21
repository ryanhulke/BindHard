# use openbabel to build molecules from the predicted geometry and atom types

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem


DEFAULT_LIGAND_ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


@dataclass
class BuildResult:
    sample_idx: int
    ok: bool
    mol: Any | None
    error: str | None = None


def decode_atomic_num(type_id: int, atom_type_decoder: dict[int, Any]) -> int:
    decoder = atom_type_decoder or {
        index: atomic_num for index, atomic_num in enumerate(DEFAULT_LIGAND_ELEMENTS)
    }
    if int(type_id) not in decoder:
        raise ValueError(
            f"unknown ligand_type index {type_id}; valid indices are {sorted(decoder)}"
        )

    atomic_num = decoder[int(type_id)]
    if isinstance(atomic_num, dict):
        atomic_num = atomic_num.get(
            "atomic_num",
            atomic_num.get("atomic_number", atomic_num.get("element")),
        )
    if isinstance(atomic_num, str):
        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atomic_num)

    atomic_num = int(atomic_num)
    if atomic_num <= 0:
        raise ValueError(f"invalid atomic number decoded from type {type_id}: {atomic_num}")
    return atomic_num


def build_mol_from_graph(
    atom_numbers: list[int],
    ligand_pos: torch.Tensor,
    bond_index: torch.Tensor,
    bond_type: torch.Tensor,
    formal_charge: torch.Tensor | None = None,
    ligand_smiles: str | None = None,
) -> Chem.Mol:
    rw_mol = Chem.RWMol()
    for atom_idx, atomic_num in enumerate(atom_numbers):
        atom = Chem.Atom(int(atomic_num))
        if formal_charge is not None:
            atom.SetFormalCharge(int(formal_charge[atom_idx].item()))
        rw_mol.AddAtom(atom)

    aromatic_atoms: set[int] = set()
    seen: set[tuple[int, int]] = set()
    for edge_idx in range(bond_index.shape[1]):
        i = int(bond_index[0, edge_idx])
        j = int(bond_index[1, edge_idx])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))

        rdkit_bond_type = BOND_TYPES.get(int(bond_type[edge_idx]))
        if rdkit_bond_type is None:
            raise ValueError(f"unknown bond type: {bond_type[edge_idx]}")
        rw_mol.AddBond(a, b, rdkit_bond_type)
        if rdkit_bond_type == Chem.BondType.AROMATIC:
            aromatic_atoms.update((a, b))

    mol = rw_mol.GetMol()
    for atom_idx in aromatic_atoms:
        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(True)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)

    if isinstance(ligand_smiles, str) and ligand_smiles.strip():
        template = Chem.MolFromSmiles(ligand_smiles)
        if template is None:
            raise ValueError(f"failed to parse ligand_smiles: {ligand_smiles}")
        match = template.GetSubstructMatch(mol)
        if match:
            mol = Chem.RenumberAtoms(template, list(match))

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, xyz in enumerate(ligand_pos.detach().cpu().float().tolist()):
        conformer.SetAtomPosition(atom_idx, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)

    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    return mol


def build_reference_mol(reference: dict[str, Any]) -> Chem.Mol:
    return build_mol_from_graph(
        atom_numbers=reference["ligand_element"].detach().cpu().long().tolist(),
        ligand_pos=reference["ligand_pos"],
        bond_index=reference["ligand_bond_index"].detach().cpu().long(),
        bond_type=reference["ligand_bond_type"].detach().cpu().long(),
        formal_charge=(
            reference["ligand_formal_charge"].detach().cpu().long()
            if "ligand_formal_charge" in reference
            else None
        ),
        ligand_smiles=reference.get("ligand_smiles"),
    )


def formal_charges_from_reference(reference: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [atom.GetFormalCharge() for atom in build_reference_mol(reference).GetAtoms()],
        dtype=torch.long,
    )


def is_valid_mol(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        test_mol = Chem.Mol(mol)
        Chem.SanitizeMol(test_mol)
        return len(Chem.GetMolFrags(test_mol)) == 1
    except Exception:
        return False


def list_target_files(inference_dir: Path) -> list[Path]:
    return sorted(
        path / "target.pt"
        for path in inference_dir.iterdir()
        if path.is_dir() and (path / "target.pt").exists()
    )


def load_target_file(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    required = {"target_idx", "protein", "reference", "samples"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return data


def infer_mol_from_geometry(
    ligand_pos: torch.Tensor,
    ligand_type: torch.Tensor,
    atom_type_decoder: dict[int, Any],
    ligand_bond_index: torch.Tensor | None = None,
    ligand_bond_type: torch.Tensor | None = None,
    ligand_charge: torch.Tensor | None = None,
    ligand_smiles: str | None = None,
) -> Any | None:
    if ligand_pos.numel() == 0 or ligand_type.numel() == 0:
        return None
    if ligand_pos.ndim != 2 or ligand_pos.shape[1] != 3:
        raise ValueError(f"ligand_pos must have shape [N, 3], got {tuple(ligand_pos.shape)}")
    if ligand_type.ndim != 1 or ligand_type.shape[0] != ligand_pos.shape[0]:
        raise ValueError(
            f"ligand_type must have shape [N] matching ligand_pos, got {tuple(ligand_type.shape)}"
        )

    atom_numbers = [
        decode_atomic_num(int(type_id), atom_type_decoder)
        for type_id in ligand_type.detach().cpu().long().tolist()
    ]
    if ligand_type.numel() > 0 and int(ligand_type.min().item()) < 0:
        raise ValueError(f"ligand_type contains negative indices: min={int(ligand_type.min().item())}")

    if ligand_bond_index is not None and ligand_bond_type is not None:
        try:
            return build_mol_from_graph(
                atom_numbers=atom_numbers,
                ligand_pos=ligand_pos,
                bond_index=ligand_bond_index.detach().cpu().long(),
                bond_type=ligand_bond_type.detach().cpu().long(),
                formal_charge=(
                    ligand_charge.detach().cpu().long() if ligand_charge is not None else None
                ),
                ligand_smiles=ligand_smiles,
            )
        except Exception:
            # Fall back to geometry-only bond perception for backwards compatibility.
            pass

    from openbabel import openbabel

    ob_mol = openbabel.OBMol()
    ob_mol.BeginModify()
    for xyz, atomic_num in zip(ligand_pos.detach().cpu().to(torch.float64).tolist(), atom_numbers):
        atom = ob_mol.NewAtom()
        atom.SetAtomicNum(int(atomic_num))
        atom.SetVector(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ob_mol.EndModify()
    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()

    conversion = openbabel.OBConversion()
    conversion.SetOutFormat("mol")
    mol_block = conversion.WriteString(ob_mol)
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


def build_sample(sample: dict[str, Any], atom_type_decoder: dict[int, Any]) -> BuildResult:
    sample_idx = int(sample.get("sample_idx", -1))
    try:
        mol = infer_mol_from_geometry(
            ligand_pos=sample["ligand_pos"],
            ligand_type=sample["ligand_type"],
            atom_type_decoder=atom_type_decoder,
            ligand_bond_index=sample.get("ligand_bond_index"),
            ligand_bond_type=sample.get("ligand_bond_type"),
            ligand_charge=sample.get("ligand_charge"),
            ligand_smiles=sample.get("ligand_smiles"),
        )
        if mol is None:
            return BuildResult(sample_idx=sample_idx, ok=False, mol=None, error="invalid_molecule")
        return BuildResult(sample_idx=sample_idx, ok=True, mol=mol)
    except Exception as exc:
        return BuildResult(sample_idx=sample_idx, ok=False, mol=None, error=str(exc))


def build_target(
    target: dict[str, Any],
    atom_type_decoder: dict[int, Any],
) -> dict[str, int | list[BuildResult] | dict[str, Any]]:
    return {
        "target_idx": int(target["target_idx"]),
        "protein": target["protein"],
        "reference": target["reference"],
        "samples": target["samples"],
        "built_samples": [build_sample(sample, atom_type_decoder) for sample in target["samples"]],
    }


def valid_results(results: list[BuildResult]) -> list[BuildResult]:
    return [result for result in results if result.ok]


def valid_fraction(results: list[BuildResult]) -> float:
    if not results:
        return 0.0
    return len(valid_results(results)) / len(results)
