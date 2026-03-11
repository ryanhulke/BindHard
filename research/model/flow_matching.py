import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

from model.common import AtomCountPrior, compose_context, sample_categorical, center_by_protein


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



class LigandFlowMatching(torch.nn.Module):
    def __init__(
        self,
        denoiser: torch.nn.Module,
        num_types: int,
        steps: int = 50,
        path_sigma: float = 0.1,
        type_loss_scale: float = 1.0,
        protein_noise_std: float = 0.1,
        hidden_dim: int = 256,
        protein_elem_vocab: int = 128,
        protein_aa_vocab: int = 32,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.k = int(num_types)
        self.t = int(steps)
        self.path_sigma = float(path_sigma)
        self.type_loss_scale = float(type_loss_scale)
        self.protein_noise_std = float(protein_noise_std)
        self.hidden_dim = int(hidden_dim)

        self.protein_elem_vocab = int(protein_elem_vocab)
        self.protein_aa_vocab = int(protein_aa_vocab)

        self.prot_elem = torch.nn.Embedding(self.protein_elem_vocab, self.hidden_dim)
        self.prot_aa = torch.nn.Embedding(self.protein_aa_vocab, self.hidden_dim)
        self.prot_bb = torch.nn.Embedding(2, self.hidden_dim)
        self.lig_type = torch.nn.Embedding(self.k, self.hidden_dim)

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.v_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.k),
        )

    def predict_flow_v0(
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

        h_protein = self.prot_elem(pe) + self.prot_aa(aa) + self.prot_bb(bb)

        t_atom = t_graph[ligand_batch].clamp(0.0, 1.0)
        t_feat = self.time_mlp(sinusoidal_time_embedding(t_atom, self.hidden_dim))
        h_ligand = self.lig_type(ligand_type_t.long()) + t_feat

        h_ctx, x_ctx, batch_ctx, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos_t,
            batch_protein=protein_batch,
            batch_ligand=ligand_batch,
        )

        out = self.denoiser(h_ctx, x_ctx, mask_ligand, batch_ctx, return_all=False)
        flow_hat = out["x"][mask_ligand] - ligand_pos_t
        v0_logits = self.v_head(out["h"][mask_ligand])
        return flow_hat, v0_logits

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
            if step != 0 and step != self.t and step % trajectory_stride != 0:
                return
            trajectory.append(
                {
                    "t": float(step) / float(self.t),
                    "ligand_pos": ligand_pos.detach().cpu(),
                    "ligand_type": vt_idx.detach().cpu(),
                }
            )

        record_frame(0)

        dt = 1.0 / float(self.t)
        for step in range(self.t):
            tau = (step + 0.5) * dt
            t_graph = torch.full((bsz,), tau, device=device, dtype=torch.float32)

            flow_hat, v0_logits = self.predict_flow_v0(batch_ctx, protein_pos, ligand_pos, vt_idx, t_graph)
            ligand_pos = ligand_pos + dt * flow_hat

            probs_clean = torch.softmax(v0_logits.float(), dim=-1)
            tau_next = float(step + 1) / float(self.t)
            probs_next = tau_next * probs_clean + (1.0 - tau_next) / self.k
            vt_idx = sample_categorical(probs_next)

            record_frame(step + 1)

        out = {
            "ligand_pos": ligand_pos,
            "ligand_type": vt_idx,
            "ligand_batch": ligand_batch,
            "ligand_counts": counts_t,
            "protein_pos": protein_pos,
            "protein_batch": protein_batch,
            "protein_element": protein_batch_dict["protein_element"].long(),
        }
        if trajectory is not None:
            out["trajectory"] = trajectory
        return out

    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = batch["protein_pos"].device
        protein_pos = batch["protein_pos"].float()
        ligand_pos = batch["ligand_pos"].float()
        protein_batch = batch["protein_batch"].long()
        ligand_batch = batch["ligand_batch"].long()
        bsz = int(protein_batch.max().item()) + 1

        protein_pos = protein_pos + self.protein_noise_std * torch.randn_like(protein_pos)
        protein_pos, ligand_pos = center_by_protein(protein_pos, ligand_pos, protein_batch, ligand_batch, bsz)

        t_graph = torch.rand((bsz,), device=device, dtype=torch.float32)
        t_atom = t_graph[ligand_batch]

        x_start = torch.randn_like(ligand_pos)
        xt = (1.0 - t_atom[:, None]) * x_start + t_atom[:, None] * ligand_pos
        if self.path_sigma > 0:
            xt = xt + self.path_sigma * torch.randn_like(xt)
        flow_target = ligand_pos - x_start

        v0_idx = batch["ligand_type"].long()
        vt_idx = q_sample_v(v0_idx, t_atom, self.k)

        flow_hat, v0_logits = self.predict_flow_v0(batch, protein_pos, xt, vt_idx, t_graph)

        loss_pos = (flow_hat.float() - flow_target.float()).pow(2).sum(dim=-1).mean()
        loss_type = F.cross_entropy(v0_logits, v0_idx) * self.type_loss_scale
        loss = loss_pos + loss_type


        return {
            "loss": loss,
            "loss_position": loss_pos,
            "loss_atom_type": loss_type,
        }
