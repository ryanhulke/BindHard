import torch
import copy

from model.flow_matching import LigandFlowMatching


class IdentityDenoiser(torch.nn.Module):
    def forward(self, h, x, mask_ligand, batch_ctx, return_all=False):
        return {"h": h, "x": x}


def _toy_batch() -> dict[str, torch.Tensor]:
    ligand_bond_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    ligand_bond_type = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    return {
        "protein_pos": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32),
        "protein_element": torch.tensor([6, 7], dtype=torch.long),
        "protein_atom_to_aa_type": torch.tensor([0, 1], dtype=torch.long),
        "protein_is_backbone": torch.tensor([0, 1], dtype=torch.long),
        "protein_batch": torch.tensor([0, 0], dtype=torch.long),
        "protein_counts": torch.tensor([2], dtype=torch.long),
        "ligand_pos": torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        "ligand_element": torch.tensor([6, 6, 8], dtype=torch.long),
        "ligand_type": torch.tensor([0, 0, 2], dtype=torch.long),
        "ligand_formal_charge": torch.tensor([0, 0, -1], dtype=torch.long),
        "ligand_batch": torch.tensor([0, 0, 0], dtype=torch.long),
        "ligand_counts": torch.tensor([3], dtype=torch.long),
        "ligand_bond_index": ligand_bond_index,
        "ligand_bond_type": ligand_bond_type,
        "ligand_bond_batch": torch.zeros(ligand_bond_type.shape[0], dtype=torch.long),
    }


def test_flow_matching_loss_reports_new_heads() -> None:
    model = LigandFlowMatching(
        denoiser=IdentityDenoiser(),
        num_types=7,
        hidden_dim=32,
        steps=4,
        max_ligand_atoms=16,
    )
    batch = _toy_batch()

    out = model.loss(batch)

    assert "loss_atom_count" in out
    assert "loss_charge" in out
    assert "loss_bond" in out
    assert torch.isfinite(out["loss"])


def test_flow_matching_sample_emits_graph_outputs() -> None:
    model = LigandFlowMatching(
        denoiser=IdentityDenoiser(),
        num_types=7,
        hidden_dim=32,
        steps=3,
        max_ligand_atoms=16,
    )
    batch = _toy_batch()
    protein_only = {
        "protein_pos": batch["protein_pos"],
        "protein_element": batch["protein_element"],
        "protein_atom_to_aa_type": batch["protein_atom_to_aa_type"],
        "protein_is_backbone": batch["protein_is_backbone"],
        "protein_batch": batch["protein_batch"],
    }

    sample = model.sample(protein_only, prior=None, num_samples=2)

    assert sample["ligand_counts"].shape == (2,)
    assert sample["ligand_charge"].shape[0] == sample["ligand_pos"].shape[0]
    assert sample["ligand_bond_index"].shape[0] == 2
    assert sample["ligand_bond_type"].ndim == 1


def test_flow_matching_nft_loss_reports_finite_value() -> None:
    model = LigandFlowMatching(
        denoiser=IdentityDenoiser(),
        num_types=7,
        hidden_dim=32,
        steps=4,
        max_ligand_atoms=16,
    )
    anchor = copy.deepcopy(model)
    batch = _toy_batch()
    reward = torch.tensor([0.75], dtype=torch.float32)

    out = model.nft_loss(
        batch,
        anchor_model=anchor,
        reward=reward,
        beta=0.3,
        beta_discrete=0.3,
    )

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_position"])
    assert torch.isfinite(out["loss_atom_type"])
