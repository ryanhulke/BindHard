from dataclasses import dataclass

@dataclass
class TrainConfig:
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
    num_types: int = 7
    norm: bool = True

    # diffusion params
    steps: int = 1000
    type_loss_scale: float = 100.0
    protein_noise_std: float = 0.1

    # train params
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    epochs: int = 10
    early_stop_patience: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0

    log_every: int = 100
    val_every: int = 1
    precision: str = "bf16" # bf16, fp16, fp32

    project: str = "bindhard"
    entity: str = "rshulke-university-of-florida"
    run_name: str = "egnn_diffusion"
    resume: bool = False
    resume_ckpt: str = "checkpoints/last.pt"
    run_id: str = None
    seed: int = 23


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
    steps: int = 1000
    type_loss_scale: float = 100.0
    protein_noise_std: float = 0.1

    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    ckpt: str = "checkpoints/best.pt"