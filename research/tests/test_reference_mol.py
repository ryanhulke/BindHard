import torch

from rdkit import Chem

from reconstruct_molecule import DEFAULT_LIGAND_ELEMENTS, build_reference_mol, build_sample


def _bond_type_to_int(bond_type: Chem.BondType) -> int:
    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    return mapping[bond_type]


def _make_reference(smiles: str, atom_order: list[int], include_smiles: bool = True) -> dict[str, torch.Tensor | str]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    ordered = Chem.RenumberAtoms(mol, atom_order)

    conf = Chem.Conformer(ordered.GetNumAtoms())
    for atom_idx in range(ordered.GetNumAtoms()):
        conf.SetAtomPosition(atom_idx, (atom_idx + 0.1, atom_idx + 0.2, atom_idx + 0.3))
    ordered.RemoveAllConformers()
    ordered.AddConformer(conf, assignId=True)

    bond_rows: list[list[int]] = [[], []]
    bond_types: list[int] = []
    for bond in ordered.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_value = _bond_type_to_int(bond.GetBondType())
        bond_rows[0].extend([begin, end])
        bond_rows[1].extend([end, begin])
        bond_types.extend([bond_value, bond_value])

    reference: dict[str, torch.Tensor | str] = {
        "ligand_element": torch.tensor(
            [atom.GetAtomicNum() for atom in ordered.GetAtoms()],
            dtype=torch.long,
        ),
        "ligand_pos": torch.tensor(
            [
                [
                    atom_idx + 0.1,
                    atom_idx + 0.2,
                    atom_idx + 0.3,
                ]
                for atom_idx in range(ordered.GetNumAtoms())
            ],
            dtype=torch.float32,
        ),
        "ligand_bond_index": torch.tensor(bond_rows, dtype=torch.long),
        "ligand_bond_type": torch.tensor(bond_types, dtype=torch.long),
    }
    if include_smiles:
        reference["ligand_smiles"] = smiles
    return reference


def test_build_reference_mol_preserves_charged_template_with_tensor_atom_order() -> None:
    reference = _make_reference(
        "O=C(O)Cc1cccc([N+](=O)[O-])c1",
        atom_order=[9, 3, 7, 1, 12, 0, 10, 4, 8, 2, 11, 5, 6],
        include_smiles=True,
    )

    mol = build_reference_mol(reference)

    assert Chem.MolToSmiles(mol) == "O=C(O)Cc1cccc([N+](=O)[O-])c1"

    formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    assert 1 in formal_charges
    assert -1 in formal_charges

    conf = mol.GetConformer()
    ligand_pos = reference["ligand_pos"]
    assert isinstance(ligand_pos, torch.Tensor)
    for atom_idx, expected in enumerate(ligand_pos.tolist()):
        observed = conf.GetAtomPosition(atom_idx)
        assert abs(observed.x - expected[0]) < 1e-6
        assert abs(observed.y - expected[1]) < 1e-6
        assert abs(observed.z - expected[2]) < 1e-6


def test_build_reference_mol_falls_back_to_graph_only_for_neutral_ligands() -> None:
    reference = _make_reference(
        "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
        atom_order=[8, 5, 10, 2, 15, 0, 14, 7, 4, 13, 1, 6, 3, 9, 11, 16, 12],
        include_smiles=False,
    )

    mol = build_reference_mol(reference)

    ligand_element = reference["ligand_element"]
    assert isinstance(ligand_element, torch.Tensor)
    assert mol.GetNumAtoms() == len(ligand_element)
    assert Chem.MolToSmiles(mol, isomericSmiles=False) == "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1"


def test_build_sample_prefers_explicit_bonds_and_charges() -> None:
    reference = _make_reference(
        "O=C(O)Cc1cccc([N+](=O)[O-])c1",
        atom_order=[9, 3, 7, 1, 12, 0, 10, 4, 8, 2, 11, 5, 6],
        include_smiles=False,
    )

    ligand_element = reference["ligand_element"]
    assert isinstance(ligand_element, torch.Tensor)
    decoder = {i: int(z) for i, z in enumerate(DEFAULT_LIGAND_ELEMENTS)}
    element_to_type = {int(z): i for i, z in decoder.items()}

    charged_reference = _make_reference(
        "O=C(O)Cc1cccc([N+](=O)[O-])c1",
        atom_order=[9, 3, 7, 1, 12, 0, 10, 4, 8, 2, 11, 5, 6],
        include_smiles=True,
    )
    charged_mol = build_reference_mol(charged_reference)
    formal_charge = torch.tensor(
        [atom.GetFormalCharge() for atom in charged_mol.GetAtoms()],
        dtype=torch.long,
    )

    sample = {
        "sample_idx": 0,
        "ligand_pos": reference["ligand_pos"],
        "ligand_type": torch.tensor(
            [element_to_type[int(z)] for z in ligand_element.tolist()],
            dtype=torch.long,
        ),
        "ligand_bond_index": reference["ligand_bond_index"],
        "ligand_bond_type": reference["ligand_bond_type"],
        "ligand_charge": formal_charge,
    }

    result = build_sample(sample, atom_type_decoder=decoder)

    assert result.ok
    assert result.mol is not None
    assert Chem.MolToSmiles(result.mol) == "O=C(O)Cc1cccc([N+](=O)[O-])c1"
