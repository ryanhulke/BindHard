# tests/diffusion_test.py
import math
import torch
import pytest

from model.diffusion import (
    center_by_protein,
    sigmoid_beta_schedule,
    cosine_alpha_sqrt_schedule,
    AtomCountPrior,
    LigandDiffusion,
)


def test_center_by_protein_centers_each_complex():
    torch.manual_seed(0)

    protein_pos = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [-5.0, 1.0, 0.0],
            [-7.0, 1.0, 0.0],
            [-6.0, 2.0, 0.0],
        ]
    )
    protein_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    ligand_pos = torch.tensor(
        [
            [11.0, 2.0, 0.0],
            [9.0, -1.0, 0.0],
            [-6.0, 2.0, 1.0],
            [-8.0, 0.0, -1.0],
        ]
    )
    ligand_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    bsz = 2
    p2, l2 = center_by_protein(protein_pos, ligand_pos, protein_batch, ligand_batch, bsz)

    mean0 = p2[protein_batch == 0].mean(dim=0)
    mean1 = p2[protein_batch == 1].mean(dim=0)
    assert torch.allclose(mean0, torch.zeros_like(mean0), atol=1e-6)
    assert torch.allclose(mean1, torch.zeros_like(mean1), atol=1e-6)

    d_before = ligand_pos[0] - ligand_pos[1]
    d_after = l2[0] - l2[1]
    assert torch.allclose(d_before, d_after, atol=1e-6)

    d_before = ligand_pos[2] - ligand_pos[3]
    d_after = l2[2] - l2[3]
    assert torch.allclose(d_before, d_after, atol=1e-6)


def test_sigmoid_beta_schedule_sane_and_monotonic():
    beta1 = 1e-7
    betaT = 2e-3
    betas = sigmoid_beta_schedule(steps=100, beta1=beta1, betaT=betaT)

    assert betas.shape == (100,)
    assert torch.all(betas > 0)
    assert torch.all(betas[1:] >= betas[:-1])

    assert float(betas.min()) >= beta1 - 1e-8
    assert float(betas.max()) <= betaT + 1e-8

    assert float(betas[0]) < beta1 + 1e-4
    assert float(betas[-1]) > betaT - 1e-4



def test_cosine_alpha_sqrt_schedule_bounds_and_trend():
    steps = 200
    a = cosine_alpha_sqrt_schedule(steps=steps, s=0.01)
    assert a.shape == (steps,)
    assert torch.all(a <= 1.0 + 1e-6)
    assert torch.all(a >= math.sqrt(0.001) - 1e-6)
    assert float(a[0]) > float(a[-1])


def test_atom_count_prior_pocket_size_simple_case():
    p = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    size = AtomCountPrior.pocket_size(p)
    assert abs(size - 5.0) < 1e-6


def test_atom_count_prior_fit_and_sample_returns_valid_count():
    torch.manual_seed(0)

    examples = []
    for i in range(20):
        p = torch.randn(30, 3) * (0.5 + 0.05 * i)
        l = torch.randn(10 + (i % 5), 3)
        examples.append({"protein_pos": p, "ligand_pos": l})

    prior = AtomCountPrior.fit(examples, n_bins=5)
    assert int(prior.edges.numel()) >= 2
    assert len(prior.values) == int(prior.edges.numel()) - 1
    assert len(prior.probs) == int(prior.edges.numel()) - 1

    test_p = torch.randn(30, 3)
    n = prior.sample(test_p, device=torch.device("cpu"))
    assert isinstance(n, int)
    assert n > 0


def test_posterior_v_is_normalized():
    torch.manual_seed(0)

    diff = LigandDiffusion(num_types=7, steps=25)
    n = 50
    vt_idx = torch.randint(0, diff.k, (n,), dtype=torch.long)
    v0_logits = torch.randn(n, diff.k)
    v0_probs = torch.softmax(v0_logits, dim=-1)
    t_atom = torch.randint(1, diff.t + 1, (n,), dtype=torch.long)

    p = diff.posterior_v(vt_idx, v0_probs, t_atom)
    assert p.shape == (n, diff.k)
    s = p.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert torch.all(p >= 0.0)


def test_loss_outputs_scalars_and_keys():
    torch.manual_seed(0)

    diff = LigandDiffusion(num_types=5, steps=30, type_loss_scale=100.0, protein_noise_std=0.1)

    protein_pos = torch.randn(12, 3)
    ligand_pos = torch.randn(7, 3)
    protein_batch = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)
    ligand_batch = torch.tensor([0] * 3 + [1] * 4, dtype=torch.long)
    ligand_type = torch.randint(0, diff.k, (7,), dtype=torch.long)

    batch = {
        "protein_pos": protein_pos,
        "ligand_pos": ligand_pos,
        "protein_batch": protein_batch,
        "ligand_batch": ligand_batch,
        "ligand_type": ligand_type,
    }

    def denoiser(batch_ctx, protein_pos_c, ligand_pos_t, ligand_type_t, t_graph):
        x0_hat = ligand_pos_t
        v0_logits = torch.zeros((ligand_pos_t.shape[0], diff.k), device=ligand_pos_t.device)
        return x0_hat, v0_logits

    out = diff.loss(denoiser, batch)
    assert set(out.keys()) == {"loss", "loss_pos", "loss_type"}
    for k in ["loss", "loss_pos", "loss_type"]:
        assert out[k].ndim == 0
        assert torch.isfinite(out[k])


def test_sample_shapes_and_consistency():
    torch.manual_seed(0)

    diff = LigandDiffusion(num_types=6, steps=15, protein_noise_std=0.1)

    examples = []
    for i in range(12):
        p = torch.randn(25, 3) * (0.8 + 0.02 * i)
        l = torch.randn(8 + (i % 4), 3)
        examples.append({"protein_pos": p, "ligand_pos": l})
    prior = AtomCountPrior.fit(examples, n_bins=4)

    protein_pos = torch.randn(40, 3)
    protein_batch = torch.tensor([0] * 20 + [1] * 20, dtype=torch.long)
    protein_batch_dict = {"protein_pos": protein_pos, "protein_batch": protein_batch}

    def denoiser(batch_ctx, protein_pos_c, ligand_pos_t, ligand_type_t, t_graph):
        x0_hat = ligand_pos_t
        v0_logits = torch.zeros((ligand_pos_t.shape[0], diff.k), device=ligand_pos_t.device)
        return x0_hat, v0_logits

    out = diff.sample(denoiser, protein_batch_dict, prior)
    assert set(out.keys()) == {
        "ligand_pos",
        "ligand_type",
        "ligand_batch",
        "ligand_counts",
        "protein_pos",
        "protein_batch",
    }
    assert out["ligand_pos"].ndim == 2 and out["ligand_pos"].shape[1] == 3
    assert out["ligand_type"].ndim == 1
    assert out["ligand_batch"].ndim == 1
    assert out["ligand_counts"].ndim == 1
    assert int(out["ligand_counts"].shape[0]) == 2
    assert int(out["ligand_pos"].shape[0]) == int(out["ligand_batch"].shape[0]) == int(out["ligand_type"].shape[0])
    assert int(out["ligand_pos"].shape[0]) == int(out["ligand_counts"].sum().item())
    assert out["ligand_type"].min().item() >= 0
    assert out["ligand_type"].max().item() < diff.k
