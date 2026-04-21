from dataclasses import dataclass


@dataclass 
class InferenceConfig:
    lmdb_path: str = "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
    split_path: str = "data/crossdocked_pose_split_from_name_val1000.pt"

    # model
    num_layers: int = 9
    hidden_dim: int = 128
    edge_feat_dim: int = 4
    num_r_gaussian: int = 16
    k: int = 32
    cutoff_mode: str = "knn"
    message_passing_mode: str = "attention" # mlp, attention
    norm: bool = True

    num_types: int = 7
    max_ligand_atoms: int = 64
    steps: int = 1000
    type_loss_scale: float = 100.0
    protein_noise_std: float = 0.1
    guidance_ckpt: str | None = None
    guidance_target: str = "vina_score"
    guidance_lower_is_better: bool = True
    guidance_scale: float = 0.0
    guidance_clip: float = 10.0

    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    ckpt: str = "checkpoints/best.pt"
