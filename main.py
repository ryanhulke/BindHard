import torch
from model.diffusion import LigandDiffusion, AtomCountPrior
from model.egnn import EGNN
from config.config import InferenceConfig
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
prior = None

@app.on_event("startup")
async def startup_event():
    global model, prior
    print(f"Loading BindHard...")
    
    with open("config/inference/base_config.yaml", "r") as f:
        cfg = InferenceConfig(**yaml.safe_load(f))
        
    denoiser = EGNN(
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        edge_feat_dim=cfg.edge_feat_dim,
        num_r_gaussian=cfg.num_r_gaussian,
        message_passing_mode=cfg.message_passing_mode,
        k=cfg.k,
        cutoff_mode=cfg.cutoff_mode,
        update_x=True
    ).to(device)

    model = LigandDiffusion(
        denoiser=denoiser,
        num_types=cfg.num_types,
        steps=cfg.steps,
        type_loss_scale=cfg.type_loss_scale,
        protein_noise_std=cfg.protein_noise_std,
    ).to(device)

    ckpt = torch.load("", map_location=device)
    model.load_state_dict(ckpt["diffusion"], strict=True)
    model.eval()
    
    prior = AtomCountPrior.from_state_dict(ckpt["prior"])
