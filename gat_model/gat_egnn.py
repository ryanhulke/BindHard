
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import softmax as pyg_softmax
from model.common import GaussianSmearing, MLP

class GATE_EQ_LAYER(nn.Module):
    """
    Args:
        hidden_dim:     Node feature dimension (must be divisible by n_heads).
        edge_feat_dim:  Bond type one-hot dimension (4 for protein/ligand edges).
        num_r_gaussian: Number of radial basis functions (passed to GaussianSmearing;
                        actual count depends on GaussianSmearing's fixed_offset mode).
        n_heads:        Number of attention heads.
        cutoff:         Distance cutoff for radial basis functions.
        update_x:       Whether to update coordinates equivariantly.
        norm:           Whether to use LayerNorm in MLPs.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        num_r_gaussian: int,
        n_heads: int = 16,
        cutoff: float = 10.0,
        update_x: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.edge_feat_dim = int(edge_feat_dim)
        self.num_r_gaussian = int(num_r_gaussian)
        self.cutoff = float(cutoff)
        self.update_x = bool(update_x)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert self.hidden_dim % self.n_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        )

        self.rbf = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_r_gaussian)
        radial_dim = int(self.rbf.offset.numel())


        if self.edge_feat_dim > 0:
            self.outer_product_dim = radial_dim * self.edge_feat_dim
        else:
            self.outer_product_dim = radial_dim

      
        kv_input_dim = self.hidden_dim + self.outer_product_dim  
        self.kv_mlp = nn.Sequential(
            nn.Linear(kv_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),  
        )

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.attn_bias_mlp = nn.Sequential(
            nn.Linear(self.outer_product_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_heads),
        )

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.node_mlp = MLP(
            in_dim=2 * self.hidden_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layer=2,
            norm=norm,
            act_fn="relu",
            act_last=False,
        )

       
        if self.update_x:
            x_mlp = [nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()]
            w = nn.Linear(self.hidden_dim, 1, bias=False)
            nn.init.xavier_uniform_(w.weight, gain=0.001)
            x_mlp += [w, nn.Tanh()]
            self.x_mlp = nn.Sequential(*x_mlp)
        else:
            self.x_mlp = None

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask_ligand: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = h.shape[0]
        num_edges = src_idx.shape[0]

        rel = x[dst_idx] - x[src_idx]                          
        d2 = (rel * rel).sum(dim=-1, keepdim=True)             
        d = torch.sqrt(d2 + 1e-8)                             

        d_feat = self.rbf(d)                                   

        if edge_attr is not None and self.edge_feat_dim > 0:
            edge_feat = (d_feat.unsqueeze(-1) * edge_attr.unsqueeze(-2)).reshape(num_edges, -1)
        else:
            edge_feat = d_feat

        kv_input = torch.cat([h[src_idx], edge_feat], dim=-1)  
        kv = self.kv_mlp(kv_input)                             
        k, v = kv.chunk(2, dim=-1)                             

        q = self.q_proj(h[dst_idx])             

        q = q.view(num_edges, self.n_heads, self.head_dim)     
        k = k.view(num_edges, self.n_heads, self.head_dim)     
        v = v.view(num_edges, self.n_heads, self.head_dim)     

        attn_logits = (q * k).sum(dim=-1) * self.scale         

        # Edge-based attention bias
        attn_bias = self.attn_bias_mlp(edge_feat)              
        attn_logits = attn_logits + attn_bias

        # Softmax over neighbors 
        attn_weights = pyg_softmax(attn_logits, dst_idx, num_nodes=num_nodes) 

        # attention-weighted values
        msg = (attn_weights.unsqueeze(-1) * v)                 
        msg = msg.reshape(num_edges, self.hidden_dim)           

        agg = torch.zeros(num_nodes, self.hidden_dim, device=msg.device, dtype=msg.dtype)
        agg.index_add_(0, dst_idx, msg)

        agg = self.out_proj(agg)                          

        h = h + self.node_mlp(torch.cat([agg, h], dim=-1))

        # Equivariant coordinate 
        if self.update_x:
            coef = self.x_mlp(msg)                             
            vec = rel / (d + 1.0) * coef                      
            dx = torch.zeros(num_nodes, x.shape[-1], device=x.device, dtype=x.dtype)
            dx.index_add_(0, dst_idx, vec.to(dtype=x.dtype))
            x = x + dx * mask_ligand.to(x.dtype).unsqueeze(-1)

        return h, x


class GATE_GNN(nn.Module):
    """
    Args:
        num_layers:     Number of equivariant attention layers.
        hidden_dim:     Node feature dimension.
        edge_feat_dim:  Bond type one-hot dimension (def=4).
        num_r_gaussian: Radial basis functions count (def=20).
        n_heads:        Number of attention heads (def=16).
        k:              k for kNN graph construction (def=32).
        cutoff:         Distance cutoff for RBF
        cutoff_mode:    Create graph using "knn" or "radius".
        update_x:       Update coords or not
        norm:           LayerNorm or not
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        edge_feat_dim: int = 4,
        num_r_gaussian: int = 20,
        n_heads: int = 16,
        k: int = 32,
        cutoff: float = 10.0,
        cutoff_mode: str = "knn",
        update_x: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.num_r_gaussian = int(num_r_gaussian)
        self.n_heads = int(n_heads)
        self.k = int(k)
        self.cutoff = float(cutoff)
        self.cutoff_mode = str(cutoff_mode)
        self.update_x = bool(update_x)

        self.layers = nn.ModuleList(
            [
                GATE_EQ_LAYER(
                    hidden_dim=self.hidden_dim,
                    edge_feat_dim=self.edge_feat_dim,
                    num_r_gaussian=self.num_r_gaussian,
                    n_heads=self.n_heads,
                    cutoff=self.cutoff,
                    update_x=self.update_x,
                    norm=norm,
                )
                for _ in range(self.num_layers)
            ]
        )

    def connect_edges(self, x: torch.Tensor, mask_ligand: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.cutoff_mode == "knn":
            return knn_graph(x, k=self.k, batch=batch, flow="source_to_target")
        if self.cutoff_mode == "radius":
            return radius_graph(x, r=self.cutoff, batch=batch, flow="source_to_target")
        raise ValueError(f"cutoff_mode must be one of: knn, radius. got {self.cutoff_mode!r}")

    def build_edge_type(self, edge_index: torch.Tensor, mask_ligand: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        ms = mask_ligand[src].bool()
        md = mask_ligand[dst].bool()
        t = torch.zeros(src.shape[0], device=edge_index.device, dtype=torch.long)
        t[ms & md] = 0
        t[ms & ~md] = 1
        t[~ms & md] = 2
        t[~ms & ~md] = 3
        return F.one_hot(t, num_classes=4).to(torch.float32)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        return_all: bool = False,
    ) -> dict:
        
        """
        Forward pass
        Args:
            h:           Node features (N, hidden_dim).
            x:           Node coordinates (N, 3).
            mask_ligand: Boolean mask, True for ligand atoms (N,).
            batch:       Batch index per node (N,).
            return_all:  If True, collect intermediate h/x from every layer.
        Returns:
            dict with keys "x", "h", "all_x", "all_h".
        """

        all_x = [x] if return_all else None
        all_h = [h] if return_all else None

        for layer in self.layers:
            edge_index = self.connect_edges(x, mask_ligand, batch)
            edge_type = self.build_edge_type(edge_index, mask_ligand)
            h, x = layer(h, x, edge_index, mask_ligand, edge_attr=edge_type)
            if return_all:
                all_x.append(x)
                all_h.append(h)

        out = {"x": x, "h": h}
        if return_all:
            out["all_x"] = all_x
            out["all_h"] = all_h
        return out
