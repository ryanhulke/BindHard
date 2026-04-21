from typing import Dict

import torch
import torch.nn.functional as F

from model.common import (
    AtomCountPrior,
    GaussianSmearing,
    center_by_protein,
    compose_context,
    q_sample_v,
    sample_categorical,
    sinusoidal_time_embedding,
)


FORMAL_CHARGE_VALUES = (-2, -1, 0, 1, 2)

def pool_by_batch(x: torch.Tensor, batch: torch.Tensor, bsz: int) -> torch.Tensor:
    if bsz == 0:
        return x.new_zeros((0, x.shape[-1]))
    out = x.new_zeros((bsz, x.shape[-1]))
    out.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=bsz).clamp_min(1).to(x.dtype)
    return out / counts[:, None]


def enumerate_ligand_pairs(ligand_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ligand_batch.numel() == 0:
        empty = ligand_batch.new_empty((0,))
        return empty, empty, empty

    bsz = int(ligand_batch.max().item()) + 1
    pair_src = []
    pair_dst = []
    pair_batch = []
    for b in range(bsz):
        idx = torch.where(ligand_batch == b)[0]
        n = int(idx.numel())
        if n < 2:
            continue
        tri = torch.triu_indices(n, n, offset=1, device=ligand_batch.device)
        pair_src.append(idx[tri[0]])
        pair_dst.append(idx[tri[1]])
        pair_batch.append(torch.full((tri.shape[1],), b, device=ligand_batch.device, dtype=torch.long))

    if not pair_src:
        empty = ligand_batch.new_empty((0,))
        return empty, empty, empty

    return torch.cat(pair_src), torch.cat(pair_dst), torch.cat(pair_batch)


def build_bond_targets(
    pair_src: torch.Tensor,
    pair_dst: torch.Tensor,
    bond_index: torch.Tensor,
    bond_type: torch.Tensor,
    n_ligand_atoms: int,
) -> torch.Tensor:
    if pair_src.numel() == 0:
        return pair_src.new_empty((0,))

    bond_map: dict[int, int] = {}
    src = bond_index[0].detach().cpu().tolist()
    dst = bond_index[1].detach().cpu().tolist()
    typ = bond_type.detach().cpu().tolist()
    for i, j, t in zip(src, dst, typ):
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = int(a) * int(n_ligand_atoms) + int(b)
        bond_map[key] = max(int(t), bond_map.get(key, 0))

    targets = [
        bond_map.get(int(a) * int(n_ligand_atoms) + int(b), 0)
        for a, b in zip(pair_src.detach().cpu().tolist(), pair_dst.detach().cpu().tolist())
    ]
    return torch.tensor(targets, device=pair_src.device, dtype=torch.long)


class LigandFlowMatching(torch.nn.Module):
    def __init__(
        self,
        denoiser: torch.nn.Module,
        num_types: int,
        steps: int = 50,
        path_sigma: float = 0.1,
        type_loss_scale: float = 4.0,
        bond_loss_scale: float = 8.0,
        charge_loss_scale: float = 1000.0,
        count_loss_scale: float = 0.5,
        protein_noise_std: float = 0.1,
        hidden_dim: int = 256,
        protein_elem_vocab: int = 128,
        protein_aa_vocab: int = 32,
        max_ligand_atoms: int = 64,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.k = int(num_types)
        self.t = int(steps)
        self.path_sigma = float(path_sigma)
        self.type_loss_scale = float(type_loss_scale)
        self.bond_loss_scale = float(bond_loss_scale)
        self.charge_loss_scale = float(charge_loss_scale)
        self.count_loss_scale = float(count_loss_scale)
        self.protein_noise_std = float(protein_noise_std)
        self.hidden_dim = int(hidden_dim)
        self.max_ligand_atoms = int(max_ligand_atoms)

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
        self.charge_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, len(FORMAL_CHARGE_VALUES)),
        )
        self.register_buffer(
            "formal_charge_values",
            torch.tensor(FORMAL_CHARGE_VALUES, dtype=torch.long),
            persistent=False,
        )

        self.bond_rbf = GaussianSmearing(start=0.0, stop=10.0, num_gaussians=16, fixed_offset=False)
        bond_in_dim = 2 * self.hidden_dim + int(self.bond_rbf.offset.numel())
        self.bond_head = torch.nn.Sequential(
            torch.nn.Linear(bond_in_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, 5),
        )

        count_in_dim = self.hidden_dim + 2
        self.count_head = torch.nn.Sequential(
            torch.nn.Linear(count_in_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, max(1, self.max_ligand_atoms)),
        )

    def _protein_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pe = batch["protein_element"].long().clamp(0, self.protein_elem_vocab - 1)
        aa = batch["protein_atom_to_aa_type"].long().clamp(0, self.protein_aa_vocab - 1)
        bb = batch["protein_is_backbone"].long().clamp(0, 1)
        return self.prot_elem(pe) + self.prot_aa(aa) + self.prot_bb(bb)

    def _atom_count_inputs(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        protein_batch = batch["protein_batch"].long()
        bsz = int(protein_batch.max().item()) + 1 if protein_batch.numel() > 0 else 0
        h_protein = self._protein_embeddings(batch)
        pooled_h = pool_by_batch(h_protein, protein_batch, bsz)
        protein_pos = batch["protein_pos"].float()
        centers = pool_by_batch(protein_pos, protein_batch, bsz)
        radius = (protein_pos - centers[protein_batch]).norm(dim=-1)
        max_radius = protein_pos.new_zeros((bsz,))
        for b in range(bsz):
            mask = protein_batch == b
            if mask.any():
                max_radius[b] = radius[mask].max()
        counts = torch.bincount(protein_batch, minlength=bsz).to(protein_pos.dtype)
        return torch.cat([pooled_h, counts[:, None], max_radius[:, None]], dim=-1), bsz

    def predict_atom_count_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        count_inputs, _ = self._atom_count_inputs(batch)
        return self.count_head(count_inputs)

    def _charge_targets(self, formal_charge: torch.Tensor) -> torch.Tensor:
        charge_values = self.formal_charge_values.to(device=formal_charge.device, dtype=formal_charge.dtype)
        return (formal_charge[:, None] - charge_values[None, :]).abs().argmin(dim=-1)

    def _pair_logits(
        self,
        ligand_h: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_src, pair_dst, pair_batch = enumerate_ligand_pairs(ligand_batch)
        if pair_src.numel() == 0:
            empty_logits = ligand_h.new_empty((0, 5))
            return pair_src, pair_dst, pair_batch, empty_logits

        rel = ligand_pos[pair_dst] - ligand_pos[pair_src]
        dist = rel.norm(dim=-1, keepdim=True)
        dist_feat = self.bond_rbf(dist)
        pair_feat = torch.cat([ligand_h[pair_src], ligand_h[pair_dst], dist_feat], dim=-1)
        return pair_src, pair_dst, pair_batch, self.bond_head(pair_feat)

    def predict_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        protein_pos: torch.Tensor,
        ligand_pos_t: torch.Tensor,
        ligand_type_t: torch.Tensor,
        t_graph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        protein_batch = batch["protein_batch"].long()
        ligand_batch = batch["ligand_batch"].long()

        h_protein = self._protein_embeddings(batch)
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
        ligand_h = out["h"][mask_ligand]
        flow_hat = out["x"][mask_ligand] - ligand_pos_t
        v0_logits = self.v_head(ligand_h)
        charge_logits = self.charge_head(ligand_h)
        pair_src, pair_dst, pair_batch, bond_logits = self._pair_logits(ligand_h, ligand_pos_t, ligand_batch)

        return {
            "flow_hat": flow_hat,
            "v0_logits": v0_logits,
            "charge_logits": charge_logits,
            "pair_src": pair_src,
            "pair_dst": pair_dst,
            "pair_batch": pair_batch,
            "bond_logits": bond_logits,
        }

    def sample_path_state(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        add_protein_noise: bool,
    ) -> Dict[str, torch.Tensor]:
        device = batch["protein_pos"].device
        protein_pos = batch["protein_pos"].float()
        ligand_pos = batch["ligand_pos"].float()
        protein_batch = batch["protein_batch"].long()
        ligand_batch = batch["ligand_batch"].long()
        bsz = int(protein_batch.max().item()) + 1 if protein_batch.numel() > 0 else 0

        if add_protein_noise and self.protein_noise_std > 0:
            protein_pos = protein_pos + self.protein_noise_std * torch.randn_like(protein_pos)
        protein_pos, ligand_pos = center_by_protein(protein_pos, ligand_pos, protein_batch, ligand_batch, bsz)

        t_graph = torch.rand((bsz,), device=device, dtype=torch.float32)
        t_atom = t_graph[ligand_batch]
        x_start = torch.randn_like(ligand_pos)
        ligand_pos_t = (1.0 - t_atom[:, None]) * x_start + t_atom[:, None] * ligand_pos
        if self.path_sigma > 0:
            ligand_pos_t = ligand_pos_t + self.path_sigma * torch.randn_like(ligand_pos_t)

        return {
            "protein_pos": protein_pos,
            "ligand_pos": ligand_pos,
            "ligand_batch": ligand_batch,
            "t_graph": t_graph,
            "t_atom": t_atom,
            "x_start": x_start,
            "ligand_pos_t": ligand_pos_t,
            "ligand_type_t": q_sample_v(batch["ligand_type"].long(), t_atom, self.k),
        }

    def predict_clean_positions(
        self,
        ligand_pos_t: torch.Tensor,
        flow_hat: torch.Tensor,
        t_graph: torch.Tensor,
        ligand_batch: torch.Tensor,
    ) -> torch.Tensor:
        t_atom = t_graph[ligand_batch].clamp(0.0, 1.0)
        return ligand_pos_t + (1.0 - t_atom[:, None]) * flow_hat

    def _repeat_protein_batch(
        self,
        protein_batch_dict: Dict[str, torch.Tensor],
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        if num_samples == 1:
            return protein_batch_dict

        protein_batch = protein_batch_dict["protein_batch"].long()
        n_atoms = int(protein_batch.numel())
        bsz = int(protein_batch.max().item()) + 1
        gather_idx = torch.cat(
            [torch.where(protein_batch == b)[0].repeat(num_samples) for b in range(bsz)],
            dim=0,
        )
        repeated_counts = torch.bincount(protein_batch, minlength=bsz).repeat_interleave(num_samples)
        repeated_batch = torch.repeat_interleave(
            torch.arange(bsz * num_samples, device=protein_batch.device, dtype=torch.long),
            repeated_counts,
        )
        out = {
            k: (v[gather_idx] if v.ndim > 0 and int(v.shape[0]) == n_atoms else v)
            for k, v in protein_batch_dict.items()
        }
        out["protein_batch"] = repeated_batch
        return out

    def _sample_counts(
        self,
        protein_batch_dict: Dict[str, torch.Tensor],
        prior: AtomCountPrior | None,
        device: torch.device,
    ) -> torch.Tensor:
        protein_batch = protein_batch_dict["protein_batch"].long()
        bsz = int(protein_batch.max().item()) + 1 if protein_batch.numel() > 0 else 0
        if prior is not None:
            protein_pos = protein_batch_dict["protein_pos"].float()
            counts = []
            for b in range(bsz):
                counts.append(prior.sample(protein_pos[protein_batch == b], device))
            return torch.tensor(counts, device=device, dtype=torch.long)

        logits = self.predict_atom_count_logits(protein_batch_dict)
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1).to(torch.long) + 1

    def _decode_structure(
        self,
        ligand_batch: torch.Tensor,
        charge_logits: torch.Tensor,
        pair_src: torch.Tensor,
        pair_dst: torch.Tensor,
        bond_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        charge_idx = charge_logits.argmax(dim=-1)
        ligand_charge = self.formal_charge_values[charge_idx].to(device=charge_logits.device)

        if pair_src.numel() == 0:
            empty_index = pair_src.new_empty((2, 0))
            empty_type = pair_src.new_empty((0,))
            return ligand_charge, empty_index, empty_type

        bond_pred = bond_logits.argmax(dim=-1)
        keep = bond_pred > 0
        if not keep.any():
            empty_index = pair_src.new_empty((2, 0))
            empty_type = pair_src.new_empty((0,))
            return ligand_charge, empty_index, empty_type

        bond_index = torch.stack([pair_src[keep], pair_dst[keep]], dim=0)
        bond_type = bond_pred[keep]
        return ligand_charge, bond_index, bond_type

    def _apply_guidance(
        self,
        guidance_model: torch.nn.Module,
        batch_ctx: Dict[str, torch.Tensor],
        protein_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_type: torch.Tensor,
        guidance_lower_is_better: bool,
        guidance_scale: float,
        guidance_clip: float,
        dt: float,
    ) -> torch.Tensor:
        with torch.enable_grad():
            pos_in = ligand_pos.detach().requires_grad_(True)
            score = guidance_model(
                batch=batch_ctx,
                protein_pos=protein_pos.detach(),
                ligand_pos=pos_in,
                ligand_type=ligand_type.detach(),
            )
            objective = -score.sum() if guidance_lower_is_better else score.sum()
            grad = torch.autograd.grad(objective, pos_in, allow_unused=False)[0]

        step = (guidance_scale * dt) * grad

        if guidance_clip > 0:
            step_norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = (guidance_clip / step_norm).clamp_max(1.0)
            step = step * scale

        return (ligand_pos - step).detach()

    def sample(
        self,
        protein_batch_dict: Dict[str, torch.Tensor],
        prior: AtomCountPrior | None = None,
        return_trajectory: bool = False,
        trajectory_stride: int = 1,
        num_samples: int = 1,
        num_steps: int | None = None,
        time_schedule: str = "legacy",
        type_temperature: float = 1.0,
        guidance_model: torch.nn.Module | None = None,
        guidance_lower_is_better: bool = True,
        guidance_scale: float = 0.0,
        guidance_clip: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        device = protein_batch_dict["protein_pos"].device
        protein_batch_dict = self._repeat_protein_batch(protein_batch_dict, int(num_samples))
        protein_pos = protein_batch_dict["protein_pos"].float()
        protein_batch = protein_batch_dict["protein_batch"].long()
        bsz = int(protein_batch.max().item()) + 1 if protein_batch.numel() > 0 else 0
        trajectory_stride = max(1, int(trajectory_stride))
        num_steps = self.t if num_steps is None else max(1, int(num_steps))
        type_temperature = max(float(type_temperature), 1e-6)

        counts_t = self._sample_counts(protein_batch_dict, prior=prior, device=device)
        ligand_batch = torch.repeat_interleave(torch.arange(bsz, device=device), counts_t)
        n_lig = int(ligand_batch.numel())

        ligand_pos = torch.randn((n_lig, 3), device=device)
        vt_idx = torch.randint(0, self.k, (n_lig,), device=device)
        protein_pos, ligand_pos = center_by_protein(protein_pos, ligand_pos, protein_batch, ligand_batch, bsz)

        batch_ctx = dict(protein_batch_dict)
        batch_ctx["ligand_batch"] = ligand_batch
        guidance_enabled = guidance_model is not None and guidance_scale > 0.0

        trajectory = [] if return_trajectory else None

        def record_frame(time_value: float, step_idx: int) -> None:
            if trajectory is None:
                return
            if step_idx != 0 and step_idx != num_steps and step_idx % trajectory_stride != 0:
                return
            trajectory.append(
                {
                    "t": float(time_value),
                    "ligand_pos": ligand_pos.detach().cpu(),
                    "ligand_type": vt_idx.detach().cpu(),
                }
            )

        record_frame(0.0, 0)

        if time_schedule == "legacy":
            dt = 1.0 / float(num_steps)
            for step in range(num_steps):
                tau = (step + 0.5) * dt
                t_graph = torch.full((bsz,), tau, device=device, dtype=torch.float32)

                out = self.predict_outputs(batch_ctx, protein_pos, ligand_pos, vt_idx, t_graph)
                ligand_pos = (ligand_pos + dt * out["flow_hat"]).detach()

                if guidance_enabled:
                    ligand_pos = self._apply_guidance(
                        guidance_model=guidance_model,
                        batch_ctx=batch_ctx,
                        protein_pos=protein_pos,
                        ligand_pos=ligand_pos,
                        ligand_type=vt_idx,
                        guidance_lower_is_better=bool(guidance_lower_is_better),
                        guidance_scale=float(guidance_scale),
                        guidance_clip=float(guidance_clip),
                        dt=dt,
                    )

                probs_clean = torch.softmax(out["v0_logits"].float() / type_temperature, dim=-1)
                tau_next = float(step + 1) / float(num_steps)
                probs_next = tau_next * probs_clean + (1.0 - tau_next) / self.k
                vt_idx = sample_categorical(probs_next)
                record_frame(tau_next, step + 1)
        else:
            if time_schedule == "linear":
                time_grid = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
            elif time_schedule == "log":
                time_grid = torch.flip(
                    1.0 - torch.logspace(-2.0, 0.0, steps=num_steps + 1, device=device),
                    dims=(0,),
                )
            else:
                raise ValueError(f"time_schedule must be one of: legacy, linear, log. got {time_schedule!r}")

            for step in range(1, num_steps + 1):
                t_prev = float(time_grid[step - 1].item())
                t_curr = float(time_grid[step].item())
                dt = t_curr - t_prev
                t_graph = torch.full((bsz,), t_curr, device=device, dtype=torch.float32)

                out = self.predict_outputs(batch_ctx, protein_pos, ligand_pos, vt_idx, t_graph)
                ligand_pos = (ligand_pos + dt * out["flow_hat"]).detach()

                if guidance_enabled:
                    ligand_pos = self._apply_guidance(
                        guidance_model=guidance_model,
                        batch_ctx=batch_ctx,
                        protein_pos=protein_pos,
                        ligand_pos=ligand_pos,
                        ligand_type=vt_idx,
                        guidance_lower_is_better=bool(guidance_lower_is_better),
                        guidance_scale=float(guidance_scale),
                        guidance_clip=float(guidance_clip),
                        dt=dt,
                    )

                probs_clean = torch.softmax(out["v0_logits"].float() / type_temperature, dim=-1)
                probs_next = t_curr * probs_clean + (1.0 - t_curr) / self.k
                vt_idx = sample_categorical(probs_next)
                record_frame(t_curr, step)

        t_final = torch.ones((bsz,), device=device, dtype=torch.float32)
        final = self.predict_outputs(batch_ctx, protein_pos, ligand_pos, vt_idx, t_final)
        ligand_charge, bond_index, bond_type = self._decode_structure(
            ligand_batch=ligand_batch,
            charge_logits=final["charge_logits"],
            pair_src=final["pair_src"],
            pair_dst=final["pair_dst"],
            bond_logits=final["bond_logits"],
        )

        out = {
            "ligand_pos": ligand_pos.detach(),
            "ligand_type": vt_idx.detach(),
            "ligand_charge": ligand_charge.detach(),
            "ligand_bond_index": bond_index.detach(),
            "ligand_bond_type": bond_type.detach(),
            "ligand_batch": ligand_batch.detach(),
            "ligand_counts": counts_t.detach(),
            "protein_pos": protein_pos.detach(),
            "protein_batch": protein_batch.detach(),
            "protein_element": protein_batch_dict["protein_element"].long().detach(),
        }
        if trajectory is not None:
            out["trajectory"] = trajectory
        return out

    def nft_loss(
        self,
        batch: Dict[str, torch.Tensor],
        anchor_model: torch.nn.Module,
        reward: torch.Tensor,
        *,
        beta: float,
        beta_discrete: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        state = self.sample_path_state(batch, add_protein_noise=False)
        out = self.predict_outputs(
            batch,
            state["protein_pos"],
            state["ligand_pos_t"],
            state["ligand_type_t"],
            state["t_graph"],
        )

        with torch.no_grad():
            anchor_out = anchor_model.predict_outputs(
                batch,
                state["protein_pos"],
                state["ligand_pos_t"],
                state["ligand_type_t"],
                state["t_graph"],
            )

        x_hat = self.predict_clean_positions(
            state["ligand_pos_t"],
            out["flow_hat"],
            state["t_graph"],
            state["ligand_batch"],
        )
        x_hat_anchor = self.predict_clean_positions(
            state["ligand_pos_t"],
            anchor_out["flow_hat"],
            state["t_graph"],
            state["ligand_batch"],
        )
        x_pos = (1.0 - float(beta)) * x_hat_anchor + float(beta) * x_hat
        x_neg = (1.0 + float(beta)) * x_hat_anchor - float(beta) * x_hat

        reward = reward.float().view(-1)
        bsz = int(reward.numel())
        if bsz == 0:
            raise ValueError("nft_loss requires at least one reward value")
        if int(state["t_graph"].numel()) != bsz:
            raise ValueError(
                f"reward shape mismatch: expected {int(state['t_graph'].numel())}, got {bsz}"
            )

        pos_err = (x_pos.float() - state["ligand_pos"].float()).pow(2).sum(dim=-1)
        neg_err = (x_neg.float() - state["ligand_pos"].float()).pow(2).sum(dim=-1)
        per_mol_pos = pool_by_batch(pos_err[:, None], state["ligand_batch"], bsz).squeeze(-1)
        per_mol_neg = pool_by_batch(neg_err[:, None], state["ligand_batch"], bsz).squeeze(-1)
        loss_pos = (reward * per_mol_pos + (1.0 - reward) * per_mol_neg).mean()

        beta_v = float(beta) if beta_discrete is None else float(beta_discrete)
        logits_pos = (1.0 - beta_v) * anchor_out["v0_logits"] + beta_v * out["v0_logits"]
        logits_neg = (1.0 + beta_v) * anchor_out["v0_logits"] - beta_v * out["v0_logits"]
        ce_pos = F.cross_entropy(logits_pos, batch["ligand_type"].long(), reduction="none")
        ce_neg = F.cross_entropy(logits_neg, batch["ligand_type"].long(), reduction="none")
        per_mol_type_pos = pool_by_batch(ce_pos[:, None], state["ligand_batch"], bsz).squeeze(-1)
        per_mol_type_neg = pool_by_batch(ce_neg[:, None], state["ligand_batch"], bsz).squeeze(-1)
        loss_type = (
            reward * per_mol_type_pos + (1.0 - reward) * per_mol_type_neg
        ).mean() * self.type_loss_scale

        loss = loss_pos + loss_type
        return {
            "loss": loss,
            "loss_position": loss_pos,
            "loss_atom_type": loss_type,
        }

    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = batch["protein_pos"].device
        state = self.sample_path_state(batch, add_protein_noise=True)
        out = self.predict_outputs(
            batch,
            state["protein_pos"],
            state["ligand_pos_t"],
            state["ligand_type_t"],
            state["t_graph"],
        )

        flow_target = state["ligand_pos"] - state["x_start"]
        loss_pos = (out["flow_hat"].float() - flow_target.float()).pow(2).sum(dim=-1).mean()
        loss_type = F.cross_entropy(out["v0_logits"], batch["ligand_type"].long()) * self.type_loss_scale

        count_logits = self.predict_atom_count_logits(batch)
        count_target = batch["ligand_counts"].long().sub(1).clamp(0, max(0, self.max_ligand_atoms - 1))
        loss_count = F.cross_entropy(count_logits, count_target) * self.count_loss_scale

        charge_target = self._charge_targets(batch["ligand_formal_charge"].long())
        loss_charge = F.cross_entropy(out["charge_logits"], charge_target) * self.charge_loss_scale

        pair_src = out["pair_src"]
        pair_dst = out["pair_dst"]
        if pair_src.numel() == 0:
            loss_bond = torch.zeros((), device=device, dtype=loss_pos.dtype)
        else:
            bond_target = build_bond_targets(
                pair_src=pair_src,
                pair_dst=pair_dst,
                bond_index=batch["ligand_bond_index"].long(),
                bond_type=batch["ligand_bond_type"].long(),
                n_ligand_atoms=int(batch["ligand_pos"].shape[0]),
            )
            loss_bond = F.cross_entropy(out["bond_logits"], bond_target) * self.bond_loss_scale

        loss = loss_pos + loss_type + loss_count + loss_charge + loss_bond

        return {
            "loss": loss,
            "loss_position": loss_pos,
            "loss_atom_type": loss_type,
            "loss_atom_count": loss_count,
            "loss_charge": loss_charge,
            "loss_bond": loss_bond,
        }
