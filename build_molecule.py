# use openbabel to build molecules from the predicted geometry and atom types

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import openbabel
from rdkit import Chem

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

# infer bonds between atoms in ligand
def infer_mol_from_geometry(ligand_pos: torch.Tensor,ligand_type: torch.Tensor, atom_type_decoder: dict[int, Any], ) -> Any | None:
    ob = openbabel.openbabel
    pos = ligand_pos.detach().cpu().float().numpy()
    types = ligand_type.detach().cpu().numpy()
    n = pos.shape[0]

    if n == 0: return None
    mol_ob = ob.OBMol()
    mol_ob.BeginModify()

    for i in range(n):
        type_idx = int(types[i])
        atomic_num = atom_type_decoder.get(type_idx)
        if atomic_num is None: return None
        atom = mol_ob.NewAtom()
        atom.SetAtomicNum(int(atomic_num))
        atom.SetVector(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))

    mol_ob.EndModify()
    mol_ob.ConnectTheDots()       # infer bonds from covalent-radius
    mol_ob.PerceiveBondOrders()   # assign bond orders
    conv = ob.OBConversion()
    conv.SetOutFormat("smi")
    smi = conv.WriteString(mol_ob).strip()

    if not smi: return None
    return Chem.MolFromSmiles(smi) # docs said to ret this?


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
) -> dict[str, Any]:
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