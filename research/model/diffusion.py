import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from model.common import AtomCountPrior, compose_context, sample_categorical, center_by_protein


#  noise schedule for the Position
def sigmoid_beta_schedule(steps: int, beta1: float, betaT: float) -> torch.Tensor:
    x = torch.linspace(-6, 6, steps, dtype=torch.float64)
    sig = 1.0 / (torch.exp(-x) + 1.0)
    return (sig * (betaT - beta1) + beta1).to(torch.float32)

#  noise schedule for the atom type
def cosine_alpha_sqrt_schedule(steps: int, s: float) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    ac = torch.cos(((x / steps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
    ac = ac / ac[0]
    a = (ac[1:] / ac[:-1]).clamp(0.001, 1.0)
    return torch.sqrt(a).to(torch.float32)



class LigandDiffusion(torch.nn.Module):
    def __init__(
        self,
        denoiser: torch.nn.Module,
        num_types: int,
        steps: int = 1000,
        type_loss_scale: float = 100.0,
        protein_noise_std: float = 0.1,
        beta1: float = 1e-7,
        betaT: float = 2e-3,
        type_cos_s: float = 0.01,
        hidden_dim: int = 256,
        protein_elem_vocab: int = 128,
        protein_aa_vocab: int = 32,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.k = int(num_types)
        self.t = int(steps)
        self.type_loss_scale = float(type_loss_scale)
        self.protein_noise_std = float(protein_noise_std)

        self.protein_elem_vocab = int(protein_elem_vocab)
        self.protein_aa_vocab = int(protein_aa_vocab)

        self.prot_elem = torch.nn.Embedding(self.protein_elem_vocab, hidden_dim)

        # amino acid type can provide some coarse information about the local chemical environment of the protein atom, 
        # which can help the model learn better representations and improve the accuracy of the predicted ligand atom types.
        self.prot_aa = torch.nn.Embedding(self.protein_aa_vocab, hidden_dim) 

        # whether the protein atom is a backbone atom
        self.prot_bb = torch.nn.Embedding(2, hidden_dim)

        # ligand atom type
        self.lig_type = torch.nn.Embedding(self.k, hidden_dim)

        # time step embedding
        self.time = torch.nn.Embedding(self.t, hidden_dim)

        self.v_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, self.k),
        )

        # noise schedule for the atom positions, 
        # precomputed and stored as buffers for efficient indexing during training and sampling
        betas = sigmoid_beta_schedule(self.t, beta1, betaT)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)

        # regsister_buffer() stores tensors that aren't learnable params but are part of module's state,
        # and move with the module across devices (to GPU/CPU) and are saved/loaded in the state_dict.
        self.register_buffer("betas_x", torch.cat([torch.zeros(1), betas], dim=0))
        self.register_buffer("alphas_x", torch.cat([torch.ones(1), alphas], dim=0))
        self.register_buffer("abar_x", torch.cat([torch.ones(1), abar], dim=0))

        # noise schedule for the position
        alpha_step_sqrt = cosine_alpha_sqrt_schedule(self.t, type_cos_s)
        abar_v = torch.cumprod(alpha_step_sqrt, dim=0)
        self.register_buffer("alpha_v", torch.cat([torch.ones(1), alpha_step_sqrt], dim=0))
        self.register_buffer("abar_v", torch.cat([torch.ones(1), abar_v], dim=0))

    # x, the atom position, is represented as a Gaussian distribution
    def q_sample_x(self, x0: torch.Tensor, t_atom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abar = self.abar_x[t_atom].unsqueeze(-1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * eps
        return xt, eps

    # v, the atom type, is represented as one-hot, so we can directly sample from the categorical distribution
    def q_sample_v(self, v0_idx: torch.Tensor, t_atom: torch.Tensor) -> torch.Tensor:
        abar = self.abar_v[t_atom].unsqueeze(-1)
        probs = abar * F.one_hot(v0_idx, self.k).to(torch.float32) + (1.0 - abar) / self.k
        return sample_categorical(probs)

    def posterior_x(self, xt: torch.Tensor, x0_hat: torch.Tensor, t_atom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.betas_x[t_atom].unsqueeze(-1)
        alpha = self.alphas_x[t_atom].unsqueeze(-1)
        abar = self.abar_x[t_atom].unsqueeze(-1)
        abar_prev = self.abar_x[(t_atom - 1).clamp_min(0)].unsqueeze(-1)
        coef1 = beta * torch.sqrt(abar_prev) / (1.0 - abar)
        coef2 = torch.sqrt(alpha) * (1.0 - abar_prev) / (1.0 - abar)
        mean = coef1 * x0_hat + coef2 * xt
        var = beta * (1.0 - abar_prev) / (1.0 - abar)
        return mean, var.clamp_min(1e-20)

    def posterior_v(self, vt_idx: torch.Tensor, v0_probs: torch.Tensor, t_atom: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_v[t_atom].unsqueeze(-1)
        abar_prev = self.abar_v[(t_atom - 1).clamp_min(0)].unsqueeze(-1)
        like = alpha * F.one_hot(vt_idx, self.k).to(torch.float32) + (1.0 - alpha) / self.k
        prior = abar_prev * v0_probs + (1.0 - abar_prev) / self.k
        p = like * prior
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    def predict_x0_v0(
        self,
        batch: Dict[str, torch.Tensor],
        protein_pos: torch.Tensor,
        ligand_pos_t: torch.Tensor,
        ligand_type_t: torch.Tensor,
        t_graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        protein_batch = batch["protein_batch"].long()
        ligand_batch = batch["ligand_batch"].long()

        pe = batch["protein_element"].long().clamp(0, self.protein_elem_vocab - 1)
        aa = batch["protein_atom_to_aa_type"].long().clamp(0, self.protein_aa_vocab - 1)
        bb = batch["protein_is_backbone"].long().clamp(0, 1)

        # protein atoms' hidden feature vectors
        # sum of the embeddings of element type, amino acid type, and whether it's a backbone atom
        h_protein = self.prot_elem(pe) + self.prot_aa(aa) + self.prot_bb(bb)

        t_atom = t_graph[ligand_batch].clamp(0, self.t - 1)

        # ligand atoms' hidden feature vectors, sum of the embedding of the atom type and the time step
        h_ligand = self.lig_type(ligand_type_t.long()) + self.time(t_atom)


        # combine the protein and ligand features and positions into a single context for the EGNN denoiser
        h_ctx, x_ctx, batch_ctx, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos_t,
            batch_protein=protein_batch,
            batch_ligand=ligand_batch,
        )

        # predict the original atom positions and types from the noisy input
        out = self.denoiser(h_ctx, x_ctx, mask_ligand, batch_ctx, return_all=False)
        x0_hat = out["x"][mask_ligand]
        v0_logits = self.v_head(out["h"][mask_ligand])
        return x0_hat, v0_logits

    # the sampling process starts from pure noise and iteratively denoises it using the learned model
    @torch.no_grad()
    def sample(
        self,
        protein_batch_dict: Dict[str, torch.Tensor],
        prior: AtomCountPrior,
        return_trajectory: bool = False,
        trajectory_stride: int = 1,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        device = protein_batch_dict["protein_pos"].device
        protein_pos = protein_batch_dict["protein_pos"].float()
        protein_batch = protein_batch_dict["protein_batch"].long()
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        bsz = int(protein_batch.max().item()) + 1
        if num_samples > 1:
            n_atoms = int(protein_batch.numel())
            gather_idx = torch.cat(
                [torch.where(protein_batch == b)[0].repeat(num_samples) for b in range(bsz)],
                dim=0,
            )
            repeated_counts = torch.bincount(protein_batch, minlength=bsz).repeat_interleave(num_samples)
            protein_batch = torch.repeat_interleave(
                torch.arange(bsz * num_samples, device=device, dtype=torch.long),
                repeated_counts,
            )
            protein_batch_dict = {
                k: (v[gather_idx] if v.ndim > 0 and int(v.shape[0]) == n_atoms else v)
                for k, v in protein_batch_dict.items()
            }
            protein_batch_dict["protein_batch"] = protein_batch
            protein_pos = protein_batch_dict["protein_pos"].float()
            bsz = int(protein_batch.max().item()) + 1
        trajectory_stride = max(1, int(trajectory_stride))

        counts = []
        for b in range(bsz):
            counts.append(prior.sample(protein_pos[protein_batch == b], device))
        counts_t = torch.tensor(counts, device=device, dtype=torch.long)

        ligand_batch = torch.repeat_interleave(torch.arange(bsz, device=device), counts_t)
        n_lig = int(ligand_batch.numel())

        ligand_pos = torch.randn((n_lig, 3), device=device)
        vt_idx = torch.randint(0, self.k, (n_lig,), device=device)

        protein_pos, ligand_pos = center_by_protein(protein_pos, ligand_pos, protein_batch, ligand_batch, bsz)

        batch_ctx = dict(protein_batch_dict)
        batch_ctx["ligand_batch"] = ligand_batch

        trajectory = [] if return_trajectory else None

        def record_frame(step: int) -> None:
            if trajectory is None:
                return
            if step != self.t and step != 0 and step % trajectory_stride != 0:
                return
            trajectory.append(
                {
                    "t": step,
                    "ligand_pos": ligand_pos.detach().cpu(),
                    "ligand_type": vt_idx.detach().cpu(),
                }
            )

        record_frame(self.t)

        for t in range(self.t, 0, -1):
            t_graph = torch.full((bsz,), t - 1, device=device, dtype=torch.long)
            t_atom = torch.full((n_lig,), t, device=device, dtype=torch.long)

            x0_hat, v0_logits = self.predict_x0_v0(batch_ctx, protein_pos, ligand_pos, vt_idx, t_graph)

            mean, var = self.posterior_x(ligand_pos, x0_hat.float(), t_atom)
            ligand_pos = mean if t == 1 else mean + torch.sqrt(var) * torch.randn_like(mean)

            v0_probs_hat = torch.softmax(v0_logits.float(), dim=-1)
            p = self.posterior_v(vt_idx, v0_probs_hat, t_atom)
            vt_idx = sample_categorical(p)

            record_frame(t - 1)

        out = {
            "ligand_pos": ligand_pos,
            "ligand_type": vt_idx,
            "ligand_batch": ligand_batch,
            "ligand_counts": counts_t,
            "protein_pos": protein_pos,
            "protein_batch": protein_batch,
        }

        if trajectory is not None:
            out["trajectory"] = trajectory

        return out

    # combine the regression loss for position and the scaled KL divergence for type prediction
    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = batch["protein_pos"].device
        protein_pos = batch["protein_pos"].float()
        ligand_pos = batch["ligand_pos"].float()
        protein_batch = batch["protein_batch"].long()
        ligand_batch = batch["ligand_batch"].long()
        bsz = int(protein_batch.max().item()) + 1

        protein_pos = protein_pos + self.protein_noise_std * torch.randn_like(protein_pos)
        protein_pos, ligand_pos = center_by_protein(protein_pos, ligand_pos, protein_batch, ligand_batch, bsz)

        t_graph = torch.randint(0, self.t, (bsz,), device=device, dtype=torch.long)
        t_atom = (t_graph + 1)[ligand_batch]

        v0_idx = batch["ligand_type"].long()
        xt, _ = self.q_sample_x(ligand_pos, t_atom)
        vt_idx = self.q_sample_v(v0_idx, t_atom)

        x0_hat, v0_logits = self.predict_x0_v0(batch, protein_pos, xt, vt_idx, t_graph)
        v0_probs_hat = torch.softmax(v0_logits.float(), dim=-1)

        p_true = self.posterior_v(vt_idx, F.one_hot(v0_idx, self.k).to(torch.float32), t_atom)
        p_pred = self.posterior_v(vt_idx, v0_probs_hat, t_atom)
        eps = 1e-20
        loss_type_scaled = (p_true * ((p_true + eps).log() - (p_pred + eps).log())).sum(dim=-1).mean() * self.type_loss_scale
        loss_pos = (ligand_pos - x0_hat.float()).pow(2).sum(dim=-1).mean()
        loss = loss_pos + loss_type_scaled
        return {"loss": loss, "loss_position": loss_pos, "loss_atom_type": loss_type_scaled}
