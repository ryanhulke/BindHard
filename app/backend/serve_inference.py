from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import modal
import torch
import yaml
from fastapi import Header, HTTPException
from openbabel import openbabel
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, QED, rdFingerprintGenerator, rdMolDescriptors
from vina import Vina

app = modal.App("bindhard-gpu-inference")

here = Path(__file__).resolve().parent
repo_root = here.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libxrender1", "libxext6", "libsm6", "libx11-6")
    .env({"PYTHONPATH": "/root"})
    .pip_install_from_requirements(str(here / "requirements.modal.txt"))
    .run_commands(
        "pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128",
        "pip install torch-geometric==2.7.0 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html",
    )
    .add_local_dir(str(here / "model"), remote_path="/root/model")
    .add_local_dir(str(here / "config"), remote_path="/root/config")
    .add_local_file(str(here / "aa_to_index.json"), remote_path="/root/aa_to_index.json")
    .add_local_file(str(here / "fpscores.pkl"), remote_path="/root/fpscores.pkl")
)

weights = modal.Volume.from_name("bindhard-model-weights", create_if_missing=True)

CONFIG_PATH = "/root/config/inference/flow_matching.yaml"
CHECKPOINT_PATH = "/models/graphAttn_flow_matching_best.pt"
AA_TO_INDEX_PATH = "/root/aa_to_index.json"
FPSCORES_PATH = "/root/fpscores.pkl"

CANONICAL_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

BACKBONE_ATOMS = ["CA", "C", "N", "O"]

ELEMENT_TO_ATOMIC_NUM = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "I": 53,
}

DEFAULT_LIGAND_ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
PERIODIC_TABLE = Chem.GetPeriodicTable()


class GenerationRequest(BaseModel):
    pdb_text: str
    samples_per_target: int = Field(default=8, ge=1, le=64)
    return_trajectory: bool = False
    trajectory_stride: int = Field(default=1, ge=1, le=100)


class TrajectoryFrame(BaseModel):
    t: float
    ligand_pos: list[list[float]]
    ligand_type: list[int]
    ligand_atomic_nums: list[int]
    bonds: list[list[int]]


class SampleResponse(BaseModel):
    sample_idx: int
    status: str
    error: str | None
    n_atoms: int | None
    ligand_pos: list[list[float]] | None
    ligand_type: list[int] | None
    ligand_atomic_nums: list[int] | None
    trajectory: list[TrajectoryFrame] | None
    smiles: str | None
    vina_score: float | None
    qed_score: float | None
    sa_score: float | None


class SummaryResponse(BaseModel):
    n_samples: int
    n_valid: int
    n_invalid: int
    vina_mean: float | None
    qed_mean: float | None
    sa_mean: float | None


class GenerationResponse(BaseModel):
    samples: list[SampleResponse]
    summary: SummaryResponse


def load_aa_to_index(path: str = AA_TO_INDEX_PATH) -> dict[str, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("aa_to_index.json must contain a JSON object")
    out: dict[str, int] = {}
    for key, value in raw.items():
        aa = str(key).strip().upper()
        if aa not in CANONICAL_RESIDUES:
            raise RuntimeError(f"invalid residue in aa_to_index.json: {aa}")
        out[aa] = int(value)
    missing = sorted(CANONICAL_RESIDUES.difference(out))
    if missing:
        raise RuntimeError(f"aa_to_index.json is missing residues: {missing}")
    return out


def load_sa_fpscores() -> dict[int, float]:
    data = pickle.load(Path(FPSCORES_PATH).open("rb"))
    fpscores: dict[int, float] = {}
    for entry in data:
        for bit_id in entry[1:]:
            fpscores[int(bit_id)] = float(entry[0])
    return fpscores


def infer_element_from_atom_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    if not letters:
        raise ValueError(f"cannot infer element from atom name '{atom_name}'")
    if letters.startswith("SE"):
        return "SE"
    return letters[0]


def parse_pocket_pdb_text(
    pdb_text: str,
    aa_to_index: dict[str, int],
    keep_hydrogens: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    protein_pos: list[list[float]] = []
    protein_element: list[int] = []
    protein_atom_to_aa_type: list[int] = []
    protein_is_backbone: list[bool] = []

    residue_atom_indices: dict[str, list[int]] = {}
    residue_mass_sum: dict[str, float] = {}
    residue_weighted_pos_sum: dict[str, torch.Tensor] = {}

    saw_model = False

    for line_no, raw_line in enumerate(pdb_text.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        record = line[0:6].strip().upper() if len(line) >= 6 else ""

        if record == "MODEL":
            saw_model = True
            continue
        if record == "ENDMDL" and saw_model:
            break
        if record != "ATOM":
            continue
        if len(line) < 54:
            raise ValueError(f"line {line_no}: ATOM record too short")

        altloc = line[16:17].strip().upper()
        if altloc not in {"", "A"}:
            continue

        resname = line[17:20].strip().upper()
        if resname not in aa_to_index:
            continue

        atom_name = line[12:16].strip().upper()
        chain = line[21:22].strip()
        res_id = line[22:26].strip()
        res_insert_id = line[26:27].strip()
        segment = line[72:76].strip() if len(line) >= 76 else ""

        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError as exc:
            raise ValueError(f"line {line_no}: invalid XYZ coordinates") from exc

        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = infer_element_from_atom_name(atom_name)
        if element not in ELEMENT_TO_ATOMIC_NUM:
            raise ValueError(f"line {line_no}: unsupported element '{element}'")

        atomic_num = ELEMENT_TO_ATOMIC_NUM[element]
        if atomic_num == 1 and not keep_hydrogens:
            continue

        atom_index = len(protein_pos)

        protein_pos.append([x, y, z])
        protein_element.append(atomic_num)
        protein_atom_to_aa_type.append(aa_to_index[resname])
        protein_is_backbone.append(atom_name in BACKBONE_ATOMS)

        residue_key = f"{chain}_{segment}_{res_id}_{res_insert_id}"
        if residue_key not in residue_atom_indices:
            residue_atom_indices[residue_key] = []
            residue_mass_sum[residue_key] = 0.0
            residue_weighted_pos_sum[residue_key] = torch.zeros(3, dtype=torch.float32)

        residue_atom_indices[residue_key].append(atom_index)

        atom_mass = float(PERIODIC_TABLE.GetAtomicWeight(int(atomic_num)))
        residue_mass_sum[residue_key] += atom_mass
        residue_weighted_pos_sum[residue_key] += atom_mass * torch.tensor([x, y, z], dtype=torch.float32)

    if not protein_pos:
        raise ValueError("no protein ATOM records parsed from pdb_text")

    residue_centers: list[torch.Tensor] = []
    for residue_key in residue_atom_indices:
        total_mass = residue_mass_sum[residue_key]
        if total_mass <= 0.0:
            continue
        residue_centers.append(residue_weighted_pos_sum[residue_key] / total_mass)

    if not residue_centers:
        raise ValueError("no residue centers could be computed from pdb_text")

    n_atoms = len(protein_pos)
    protein_dict = {
        "protein_pos": torch.tensor(protein_pos, dtype=torch.float32),
        "protein_batch": torch.zeros(n_atoms, dtype=torch.long),
        "protein_element": torch.tensor(protein_element, dtype=torch.long),
        "protein_atom_to_aa_type": torch.tensor(protein_atom_to_aa_type, dtype=torch.long),
        "protein_is_backbone": torch.tensor(protein_is_backbone, dtype=torch.bool),
    }
    residue_centers_tensor = torch.stack(residue_centers, dim=0)
    return protein_dict, residue_centers_tensor

def compute_box_from_pocket_atoms(protein: dict[str, torch.Tensor]) -> tuple[list[float], list[float]]:
    protein_pos = protein["protein_pos"].detach().cpu().float()
    if protein_pos.ndim != 2 or protein_pos.shape[1] != 3 or protein_pos.shape[0] == 0:
        raise ValueError("protein_pos must have shape [N, 3] with N > 0")

    mins = protein_pos.min(dim=0).values
    maxs = protein_pos.max(dim=0).values

    center = ((mins + maxs) / 2.0).tolist()
    span = (maxs - mins).tolist()

    box_size = [max(22.0, float(v) + 12.0) for v in span]
    return center, box_size

def decode_atomic_num(type_id: int, atom_type_decoder: dict[int, Any]) -> int:
    decoder = atom_type_decoder or {
        i: atomic_num for i, atomic_num in enumerate(DEFAULT_LIGAND_ELEMENTS)
    }
    if int(type_id) not in decoder:
        valid = sorted(int(x) for x in decoder.keys())
        raise ValueError(f"unknown ligand_type index {type_id}; valid indices are {valid}")

    value = decoder[int(type_id)]
    if isinstance(value, dict):
        atomic_num = value.get("atomic_num", value.get("atomic_number", value.get("element")))
    else:
        atomic_num = value

    if isinstance(atomic_num, str):
        atomic_num = PERIODIC_TABLE.GetAtomicNumber(atomic_num)
    atomic_num = int(atomic_num)
    if atomic_num <= 0:
        raise ValueError(f"invalid atomic number decoded from type {type_id}: {atomic_num}")
    return atomic_num


def build_ob_mol_from_geometry(
    ligand_pos: Any,
    ligand_type: Any,
    atom_type_decoder: dict[int, Any],
) -> openbabel.OBMol | None:
    if hasattr(ligand_pos, "detach"):
        ligand_pos = ligand_pos.detach()
    if hasattr(ligand_pos, "cpu"):
        ligand_pos = ligand_pos.cpu()
    if hasattr(ligand_pos, "tolist"):
        ligand_pos = ligand_pos.tolist()

    if hasattr(ligand_type, "detach"):
        ligand_type = ligand_type.detach()
    if hasattr(ligand_type, "cpu"):
        ligand_type = ligand_type.cpu()
    if hasattr(ligand_type, "tolist"):
        ligand_type = ligand_type.tolist()

    coords = [list(row) for row in ligand_pos]
    type_ids = [int(x) for x in ligand_type]

    if not coords or not type_ids:
        return None
    if len(coords) != len(type_ids):
        raise ValueError("ligand_pos and ligand_type must have same length")
    if not all(isinstance(row, (list, tuple)) and len(row) == 3 for row in coords):
        raise ValueError("ligand_pos must have shape [N, 3]")

    ob_mol = openbabel.OBMol()
    ob_mol.BeginModify()
    for xyz, type_id in zip(coords, type_ids):
        atom = ob_mol.NewAtom()
        atom.SetAtomicNum(decode_atomic_num(type_id, atom_type_decoder))
        atom.SetVector(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ob_mol.EndModify()
    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()
    return ob_mol


def infer_bonds_from_geometry(
    ligand_pos: Any,
    ligand_type: Any,
    atom_type_decoder: dict[int, Any],
) -> list[list[int]]:
    ob_mol = build_ob_mol_from_geometry(
        ligand_pos=ligand_pos,
        ligand_type=ligand_type,
        atom_type_decoder=atom_type_decoder,
    )
    if ob_mol is None:
        return []

    pair_to_order: dict[tuple[int, int], int] = {}
    for ob_bond in openbabel.OBMolBondIter(ob_mol):
        begin = int(ob_bond.GetBeginAtomIdx()) - 1
        end = int(ob_bond.GetEndAtomIdx()) - 1
        if begin < 0 or end < 0 or begin == end:
            continue
        a, b = (begin, end) if begin < end else (end, begin)
        order = max(1, int(ob_bond.GetBondOrder()))
        if (a, b) not in pair_to_order or order > pair_to_order[(a, b)]:
            pair_to_order[(a, b)] = order

    return [[a, b, order] for (a, b), order in sorted(pair_to_order.items())]


def infer_mol_from_geometry(
    ligand_pos: Any,
    ligand_type: Any,
    atom_type_decoder: dict[int, Any],
) -> Chem.Mol | None:
    ob_mol = build_ob_mol_from_geometry(
        ligand_pos=ligand_pos,
        ligand_type=ligand_type,
        atom_type_decoder=atom_type_decoder,
    )
    if ob_mol is None:
        return None

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


def is_usable_mol(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        test_mol = Chem.Mol(mol)
        Chem.SanitizeMol(test_mol)
        if len(Chem.GetMolFrags(test_mol)) != 1:
            return False
        return True
    except Exception:
        return False


def protein_to_pdbqt_string(protein: dict[str, torch.Tensor]) -> str:
    protein_pos = protein["protein_pos"].detach().cpu().float()
    protein_element = protein["protein_element"].detach().cpu().long()
    if "protein_batch" in protein:
        mask = protein["protein_batch"].detach().cpu().long() == 0
        protein_pos = protein_pos[mask]
        protein_element = protein_element[mask]

    ob_rec = openbabel.OBMol()
    ob_rec.BeginModify()
    for xyz, z in zip(protein_pos.tolist(), protein_element.tolist()):
        atom = ob_rec.NewAtom()
        atom.SetAtomicNum(int(z))
        atom.SetVector(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ob_rec.EndModify()

    conv = openbabel.OBConversion()
    conv.SetOutFormat("pdbqt")
    conv.AddOption("r", openbabel.OBConversion.OUTOPTIONS)
    return conv.WriteString(ob_rec)


def mol_to_pdbqt_string(mol: Chem.Mol) -> str:
    work_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(work_mol)
    work_mol = Chem.AddHs(work_mol, addCoords=True)

    if work_mol.GetNumConformers() == 0:
        status = AllChem.EmbedMolecule(work_mol, randomSeed=0)
        if status != 0:
            raise ValueError("rdkit embedding failed")

    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("mol", "pdbqt")
    ob_mol = openbabel.OBMol()
    ok = conv.ReadString(ob_mol, Chem.MolToMolBlock(work_mol))
    if not ok:
        raise ValueError("openbabel failed to read ligand mol block")
    return conv.WriteString(ob_mol)


def compute_qed(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))


def compute_sa(mol: Chem.Mol, fpscores: dict[int, float]) -> float:
    sa_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sparse_fp = sa_mfpgen.GetSparseCountFingerprint(mol)
    nze = sparse_fp.GetNonzeroElements()

    score1 = 0.0
    nf = 0
    for bit_id, count in nze.items():
        nf += count
        score1 += fpscores.get(int(bit_id), -4.0) * int(count)

    if nf == 0:
        return float("nan")

    score1 /= nf

    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms ** 1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = (
        -size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    score3 = 0.0
    num_bits = len(nze)
    if n_atoms > num_bits and num_bits > 0:
        score3 = math.log(float(n_atoms) / num_bits) * 0.5

    sa_score = score1 + score2 + score3

    raw_min = -4.0
    raw_max = 2.5
    sa_score = 11.0 - (sa_score - raw_min + 1.0) / (raw_max - raw_min) * 9.0

    if sa_score > 8.0:
        sa_score = 8.0 + math.log(sa_score - 8.0)

    if sa_score > 10.0:
        return 10.0
    if sa_score < 1.0:
        return 1.0
    return float(sa_score)


def smiles_from_mol(mol: Chem.Mol) -> str:
    mol_no_h = Chem.RemoveHs(Chem.Mol(mol))
    return str(Chem.MolToSmiles(mol_no_h, canonical=True))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


@app.cls(
    image=image,
    gpu=["L4", "A10", "T4"],
    timeout=60 * 10,
    scaledown_window=60, # scale down if no requests in the last 60 seconds
    volumes={"/models": weights},
    secrets=[modal.Secret.from_name("bindhard-inference")],
)
class LigandGenerator:
    @modal.enter()
    def load(self) -> None:
        from config.config import InferenceConfig
        from model.common import AtomCountPrior
        from model.egnn import EGNN
        from model.flow_matching import LigandFlowMatching

        with Path(CONFIG_PATH).open("r", encoding="utf-8") as f:
            cfg = InferenceConfig(**yaml.safe_load(f))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        model_kwargs: dict[str, Any] = {
            "denoiser": denoiser,
            "hidden_dim": cfg.hidden_dim,
            "num_types": cfg.num_types,
            "steps": cfg.steps,
        }
        if hasattr(cfg, "type_loss_scale"):
            model_kwargs["type_loss_scale"] = cfg.type_loss_scale
        if hasattr(cfg, "protein_noise_std"):
            model_kwargs["protein_noise_std"] = cfg.protein_noise_std

        model = LigandFlowMatching(**model_kwargs).to(device)

        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["diffusion"], strict=True)
        model.eval()

        if "prior" not in ckpt:
            raise RuntimeError("checkpoint must include 'prior' for inference")

        self.cfg = cfg
        self.device = device
        self.model = model
        self.prior = AtomCountPrior.from_state_dict(ckpt["prior"])
        self.aa_to_index = load_aa_to_index()
        self.fpscores = load_sa_fpscores()
        self.atom_type_decoder = {
            i: int(z)
            for i, z in enumerate(DEFAULT_LIGAND_ELEMENTS[:cfg.num_types])
        }

    @modal.fastapi_endpoint(method="POST")
    def generate(
        self,
        request: GenerationRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        expected = os.environ.get("INFER_BEARER_TOKEN", "").strip()
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="unauthorized")

        try:
            protein_raw_cpu, _ = parse_pocket_pdb_text(
                pdb_text=request.pdb_text,
                aa_to_index=self.aa_to_index,
                keep_hydrogens=False,
            )

            protein_shift = protein_raw_cpu["protein_pos"].mean(dim=0)

            protein_model_cpu = {
                key: value.clone()
                for key, value in protein_raw_cpu.items()
            }
            protein_model_cpu["protein_pos"] = protein_model_cpu["protein_pos"] - protein_shift

            box_center, box_size = compute_box_from_pocket_atoms(protein_raw_cpu)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        protein_gpu = {
            key: value.to(self.device, non_blocking=True)
            for key, value in protein_model_cpu.items()
        }

        amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if amp else torch.float32

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=amp,
        ):
            sample = self.model.sample(
                protein_gpu,
                self.prior,
                num_samples=request.samples_per_target,
                return_trajectory=request.return_trajectory,
                trajectory_stride=request.trajectory_stride,
            )

        ligand_batch = sample["ligand_batch"].detach().cpu().long()
        ligand_pos = sample["ligand_pos"].detach().cpu() + protein_shift
        ligand_type = sample["ligand_type"].detach().cpu().long()
        traj = sample.get("trajectory")

        receptor_pdbqt = protein_to_pdbqt_string(protein_raw_cpu)
        out_samples: list[SampleResponse] = []
        valid_vina: list[float] = []
        valid_qed: list[float] = []
        valid_sa: list[float] = []

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            receptor_path = temp_dir / "receptor.pdbqt"
            receptor_path.write_text(receptor_pdbqt, encoding="utf-8")

            vina_engine = Vina(sf_name="vina")
            vina_engine.set_receptor(str(receptor_path))
            vina_engine.compute_vina_maps(
                center=[float(x) for x in box_center],
                box_size=[float(x) for x in box_size],
            )

            for local_idx in range(request.samples_per_target):
                lig_mask = ligand_batch == local_idx
                sample_pos_tensor = ligand_pos[lig_mask]
                sample_type_tensor = ligand_type[lig_mask]
                sample_atomic_nums = [
                    decode_atomic_num(int(type_id), self.atom_type_decoder)
                    for type_id in sample_type_tensor.tolist()
                ]
                try:
                    mol = infer_mol_from_geometry(
                        ligand_pos=sample_pos_tensor,
                        ligand_type=sample_type_tensor,
                        atom_type_decoder=self.atom_type_decoder,
                    )
                    if not is_usable_mol(mol):
                        raise ValueError("invalid_rdkit_mol")

                    mol = Chem.Mol(mol)
                    Chem.SanitizeMol(mol)

                    smiles = smiles_from_mol(mol)

                    ligand_pdbqt_path = temp_dir / f"ligand_{local_idx:04d}.pdbqt"
                    ligand_pdbqt_path.write_text(mol_to_pdbqt_string(mol), encoding="utf-8")
                    vina_engine.set_ligand_from_file(str(ligand_pdbqt_path))
                    vina_score = float(vina_engine.score()[0])

                    qed_score = compute_qed(mol)
                    sa_score = compute_sa(mol, self.fpscores)

                    valid_vina.append(vina_score)
                    valid_qed.append(qed_score)
                    valid_sa.append(sa_score)

                    trajectory_payload: list[TrajectoryFrame] | None = None
                    if traj is not None:
                        trajectory_payload = []
                        for frame in traj:
                            frame_pos = frame["ligand_pos"][lig_mask].detach().cpu() + protein_shift
                            frame_type = frame["ligand_type"][lig_mask].detach().cpu().long()
                            frame_atomic_nums = [
                                decode_atomic_num(int(type_id), self.atom_type_decoder)
                                for type_id in frame_type.tolist()
                            ]
                            trajectory_payload.append(
                                TrajectoryFrame(
                                    t=float(frame["t"]),
                                    ligand_pos=frame_pos.tolist(),
                                    ligand_type=frame_type.tolist(),
                                    ligand_atomic_nums=frame_atomic_nums,
                                    bonds=infer_bonds_from_geometry(
                                        ligand_pos=frame_pos,
                                        ligand_type=frame_type,
                                        atom_type_decoder=self.atom_type_decoder,
                                    ),
                                )
                            )

                    out_samples.append(
                        SampleResponse(
                            sample_idx=local_idx,
                            status="completed",
                            error=None,
                            n_atoms=int(sample_type_tensor.shape[0]),
                            ligand_pos=sample_pos_tensor.tolist(),
                            ligand_type=sample_type_tensor.tolist(),
                            ligand_atomic_nums=sample_atomic_nums,
                            trajectory=trajectory_payload,
                            smiles=smiles,
                            vina_score=vina_score,
                            qed_score=qed_score,
                            sa_score=sa_score,
                        )
                    )
                except Exception as exc:
                    out_samples.append(
                        SampleResponse(
                            sample_idx=local_idx,
                            status="failed",
                            error=str(exc),
                            n_atoms=None,
                            ligand_pos=None,
                            ligand_type=None,
                            ligand_atomic_nums=None,
                            trajectory=None,
                            smiles=None,
                            vina_score=None,
                            qed_score=None,
                            sa_score=None,
                        )
                    )

        summary = SummaryResponse(
            n_samples=len(out_samples),
            n_valid=sum(1 for row in out_samples if row.status == "completed"),
            n_invalid=sum(1 for row in out_samples if row.status != "completed"),
            vina_mean=mean_or_none(valid_vina),
            qed_mean=mean_or_none(valid_qed),
            sa_mean=mean_or_none(valid_sa),
        )

        return GenerationResponse(samples=out_samples, summary=summary).model_dump()