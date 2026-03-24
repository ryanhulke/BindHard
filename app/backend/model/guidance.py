from typing import Dict

import torch

from model.common import compose_context


def pool_by_batch(x: torch.Tensor, batch: torch.Tensor, bsz: int) -> torch.Tensor:
    if bsz == 0:
        return x.new_zeros((0, x.shape[-1]))
    out = x.new_zeros((bsz, x.shape[-1]))
    out.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=bsz).clamp_min(1).to(x.dtype)
    return out / counts[:, None]


class PocketAffinityGuidance(torch.nn.Module):
    def __init__(
        self,
        denoiser: torch.nn.Module,
        num_types: int,
        hidden_dim: int = 256,
        protein_elem_vocab: int = 128,
        protein_aa_vocab: int = 32,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.k = int(num_types)
        self.hidden_dim = int(hidden_dim)
        self.protein_elem_vocab = int(protein_elem_vocab)
        self.protein_aa_vocab = int(protein_aa_vocab)

        self.prot_elem = torch.nn.Embedding(self.protein_elem_vocab, self.hidden_dim)
        self.prot_aa = torch.nn.Embedding(self.protein_aa_vocab, self.hidden_dim)
        self.prot_bb = torch.nn.Embedding(2, self.hidden_dim)
        self.lig_type = torch.nn.Embedding(self.k, self.hidden_dim)

        self.affinity_head = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def _protein_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pe = batch["protein_element"].long().clamp(0, self.protein_elem_vocab - 1)
        aa = batch["protein_atom_to_aa_type"].long().clamp(0, self.protein_aa_vocab - 1)
        bb = batch["protein_is_backbone"].long().clamp(0, 1)
        return self.prot_elem(pe) + self.prot_aa(aa) + self.prot_bb(bb)

    def _contact_features(
        self,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        bsz: int,
    ) -> torch.Tensor:
        feats = []
        for b in range(bsz):
            prot = protein_pos[protein_batch == b]
            lig = ligand_pos[ligand_batch == b]
            if prot.numel() == 0 or lig.numel() == 0:
                feats.append(protein_pos.new_zeros((2,)))
                continue
            dist = torch.cdist(lig.unsqueeze(0), prot.unsqueeze(0)).squeeze(0)
            min_lp = dist.min(dim=1).values
            feats.append(torch.stack([min_lp.mean(), min_lp.min()]))
        return torch.stack(feats, dim=0) if feats else protein_pos.new_zeros((0, 2))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        protein_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_type: torch.Tensor,
    ) -> torch.Tensor:
        protein_batch = batch["protein_batch"].long()
        ligand_batch = batch["ligand_batch"].long()
        bsz = int(protein_batch.max().item()) + 1 if protein_batch.numel() > 0 else 0

        h_protein = self._protein_embeddings(batch)
        h_ligand = self.lig_type(ligand_type.long())

        h_ctx, x_ctx, batch_ctx, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=protein_batch,
            batch_ligand=ligand_batch,
        )
        out = self.denoiser(h_ctx, x_ctx, mask_ligand, batch_ctx, return_all=False)
        ligand_h = out["h"][mask_ligand]
        protein_h = out["h"][~mask_ligand]

        pooled_protein = pool_by_batch(protein_h, protein_batch, bsz)
        pooled_ligand = pool_by_batch(ligand_h, ligand_batch, bsz)
        contact = self._contact_features(protein_pos, protein_batch, ligand_pos, ligand_batch, bsz)
        graph_feat = torch.cat([pooled_protein, pooled_ligand, contact], dim=-1)
        return self.affinity_head(graph_feat).squeeze(-1)