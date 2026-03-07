import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU() if act_fn == 'silu' else nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_ligand

def center_by_protein(
    protein_pos: torch.Tensor,
    ligand_pos: torch.Tensor,
    protein_batch: torch.Tensor,
    ligand_batch: torch.Tensor,
    bsz: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros((bsz, protein_pos.shape[-1]), device=protein_pos.device, dtype=protein_pos.dtype)
    sums.index_add_(0, protein_batch, protein_pos)
    counts = torch.bincount(protein_batch, minlength=bsz).clamp_min(1).to(protein_pos.dtype)
    mean_p = sums / counts[:, None]
    return protein_pos - mean_p[protein_batch], ligand_pos - mean_p[ligand_batch]


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-20
    u = torch.rand_like(probs).clamp_(eps, 1.0 - eps)
    g = -torch.log(-torch.log(u))
    return (torch.log(probs.clamp_min(eps)) + g).argmax(dim=-1)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t[:, None]
    freq = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    ang = (2.0 * math.pi) * t[:, None] * freq[None]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def q_sample_v(v0_idx: torch.Tensor, t_atom: torch.Tensor, num_types: int) -> torch.Tensor:
    keep = t_atom.unsqueeze(-1)
    probs = keep * F.one_hot(v0_idx, num_types).to(torch.float32) + (1.0 - keep) / num_types
    return sample_categorical(probs)


# before training the diffusion model, we can fit a simple prior distribution of
# ligand atom counts conditioned on the protein pocket size, which can be used to guide the sampling process.
@dataclass
class AtomCountPrior:
    edges: torch.Tensor
    values: Tuple[torch.Tensor, ...]
    probs: Tuple[torch.Tensor, ...]

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
        values = []
        probs = []
        for b in range(int(edges.numel()) - 1):
            idx = (bins == b).nonzero(as_tuple=False).flatten()
            if int(idx.numel()) == 0:
                med = int(counts_t.median().item())
                values.append(torch.tensor([med], dtype=torch.long))
                probs.append(torch.tensor([1.0], dtype=torch.float32))
                continue
            c = counts_t[idx]
            u, f = torch.unique(c, return_counts=True)
            values.append(u)
            probs.append((f.float() / f.sum()).float())

        return AtomCountPrior(edges=edges, values=tuple(values), probs=tuple(probs))

    def sample(self, protein_pos: torch.Tensor, device: torch.device) -> int:
        size = AtomCountPrior.pocket_size(protein_pos.detach().to("cpu"))
        edges = self.edges.detach().to("cpu")
        b = int(torch.bucketize(torch.tensor([size]), edges[1:-1], right=False).item())
        v = self.values[b].to(device)
        p = self.probs[b].to(device)
        j = int(torch.multinomial(p, 1).item())
        return int(v[j].item())
