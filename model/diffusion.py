import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import torch
import torch.nn.functional as F
from model.common import compose_context

def center_by_protein(
    protein_pos: torch.Tensor,
    ligand_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_batch: torch.Tensor,
    bsz: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros((bsz, protein_pos.shape[-1]), device=protein_pos.device, dtype=protein_pos.dtype)
    out.index_add_(0, protein_batch, protein_pos)
    counts = torch.bincount(protein_batch, minlength=bsz).clamp_min(1).to(protein_pos.dtype)
    mean_p = out / counts[:, None]
    return protein_pos - mean_p[protein_batch], ligand_pos - mean_p[ligand_batch]


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

# sample atom types
def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-20
    u = torch.rand_like(probs).clamp_(eps, 1.0 - eps)
    g = -torch.log(-torch.log(u))
    return (torch.log(probs.clamp_min(eps)) + g).argmax(dim=-1)


# before training the diffusion model, we can fit a simple prior distribution of
# ligand atom counts conditioned on the protein pocket size, which can be used to guide the sampling process.
@dataclass
class AtomCountPrior:
    edges: torch.Tensor
    values: Tuple[torch.Tensor, ...] # list of unique atom counts in each pocket size bin
    probs: Tuple[torch.Tensor, ...] # list of probabilities corresponding to the unique atom counts in each pocket size bin

    def state_dict(self) -> dict:
        return {
            "edges": self.edges.detach().cpu(),
            "values": [v.detach().cpu() for v in self.values],
            "probs": [p.detach().cpu() for p in self.probs],
        }

    @staticmethod
    def from_state_dict(d: dict) -> "AtomCountPrior":
        return AtomCountPrior(
            edges=d["edges"],
            values=tuple(d["values"]),
            probs=tuple(d["probs"]),
        )
    # pocket size defined as median of the top-10 farthest pairwise distances between protein pocket atoms.
    @staticmethod
    def pocket_size(protein_pos: torch.Tensor) -> float:
        n = int(protein_pos.shape[0])
        if n < 2:
            return 0.0
        d = torch.pdist(protein_pos.float(), p=2)
        k = min(10, int(d.numel()))
        top = torch.topk(d, k=k, largest=True).values
        return float(top.median().item())

    @staticmethod
    def fit(examples: Iterable[Dict[str, torch.Tensor]], n_bins: int = 10) -> "AtomCountPrior":
        sizes = []
        counts = []
        for ex in examples:
            sizes.append(AtomCountPrior.pocket_size(ex["protein_pos"]))
            counts.append(int(ex["ligand_pos"].shape[0]))
        sizes_t = torch.tensor(sizes, dtype=torch.float32)
        counts_t = torch.tensor(counts, dtype=torch.long)

        qs = torch.linspace(0, 1, n_bins + 1)
        edges = torch.quantile(sizes_t, qs).unique(sorted=True)
        if int(edges.numel()) < 2:
            mn = float(sizes_t.min().item())
            mx = float(sizes_t.max().item())
            edges = torch.tensor([mn, mx + 1e-3], dtype=torch.float32)

        bins = torch.bucketize(sizes_t, edges[1:-1], right=False)
        vals = []
        probs = []
        for b in range(int(edges.numel()) - 1):
            idx = (bins == b).nonzero(as_tuple=False).flatten()
            if int(idx.numel()) == 0:
                med = int(counts_t.median().item())
                vals.append(torch.tensor([med], dtype=torch.long))
                probs.append(torch.tensor([1.0], dtype=torch.float32))
                continue
            c = counts_t[idx]
            u, f = torch.unique(c, return_counts=True)
            vals.append(u)
            probs.append((f.float() / f.sum()).float())
        return AtomCountPrior(edges=edges, values=tuple(vals), probs=tuple(probs))

    def sample(self, protein_pos: torch.Tensor, device: torch.device) -> int:
        size = AtomCountPrior.pocket_size(protein_pos.detach().to("cpu"))
        edges = self.edges.detach().to("cpu")
        b = int(torch.bucketize(torch.tensor([size]), edges[1:-1], right=False).item())
        v = self.values[b].to(device)
        p = self.probs[b].to(device)
        j = int(torch.multinomial(p, 1).item())
        return int(v[j].item())


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
    def sample(self, protein_batch_dict: Dict[str, torch.Tensor], prior: AtomCountPrior) -> Dict[str, torch.Tensor]:
        device = protein_batch_dict["protein_pos"].device
        protein_pos = protein_batch_dict["protein_pos"].float()
        protein_batch = protein_batch_dict["protein_batch"].long()
        bsz = int(protein_batch.max().item()) + 1

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

        for t in range(self.t, 0, -1):
            t_graph = torch.full((bsz,), t - 1, device=device, dtype=torch.long)
            t_atom = torch.full((n_lig,), t, device=device, dtype=torch.long)

            # predict the original atom positions and types from the noisy input
            x0_hat, v0_logits = self.predict_x0_v0(batch_ctx, protein_pos, ligand_pos, vt_idx, t_graph)

            # interpolate between the predicted mean x0_hat and the noisy xt according to variance schedule
            mean, var = self.posterior_x(ligand_pos, x0_hat.float(), t_atom)
            ligand_pos = mean if t == 1 else mean + torch.sqrt(var) * torch.randn_like(mean)

            v0_probs_hat = torch.softmax(v0_logits.float(), dim=-1)
            p = self.posterior_v(vt_idx, v0_probs_hat, t_atom)
            vt_idx = sample_categorical(p)

        return {
            "ligand_pos": ligand_pos,
            "ligand_type": vt_idx,
            "ligand_batch": ligand_batch,
            "ligand_counts": counts_t,
            "protein_pos": protein_pos,
            "protein_batch": protein_batch,
        }

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