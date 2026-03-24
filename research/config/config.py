from dataclasses import dataclass


@dataclass
class BaseConfig:
    lmdb_path: str = "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
    split_path: str = "data/crossdocked_pose_split_from_name_val1000.pt"

    # model
    num_layers: int = 9
    hidden_dim: int = 128
    edge_feat_dim: int = 4
    num_r_gaussian: int = 16
    k: int = 32
    cutoff_mode: str = "knn"
    message_passing_mode: str = "attention"  # mlp, attention
    norm: bool = True
    num_types: int = 7

    # loader
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None


@dataclass
class TrainConfig(BaseConfig):
    # diffusion params
    max_ligand_atoms: int = 64
    steps: int = 1000
    type_loss_scale: float = 100.0
    bond_loss_scale: float = 1.0
    charge_loss_scale: float = 0.5
    count_loss_scale: float = 0.5
    protein_noise_std: float = 0.1

    # train params
    epochs: int = 10
    early_stop_patience: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-6
    optimizer: str = "adamw"
    grad_clip: float = 1.0
    log_every: int = 100
    val_every: int = 1
    precision: str = "bf16"  # bf16, fp16, fp32

    project: str = "bindhard"
    entity: str = "rshulke-university-of-florida"
    run_name: str = "egnn_diffusion"
    resume: bool = False
    resume_ckpt: str = "checkpoints/last.pt"
    run_id: str | None = None
    seed: int = 23


@dataclass
class InferenceConfig(BaseConfig):
    max_ligand_atoms: int = 64
    steps: int = 1000
    type_loss_scale: float = 100.0
    protein_noise_std: float = 0.1
    guidance_ckpt: str | None = None
    guidance_target: str = "vina_score"
    guidance_lower_is_better: bool = True
    guidance_scale: float = 0.0
    guidance_clip: float = 10.0
    samples_per_target: int = 100
    sample_batch_size: int = 50
    save_trajectory: bool = True
    ckpt: str = "checkpoints/best.pt"


@dataclass
class GuidanceConfig(BaseConfig):
    guidance_target: str = "vina_score"
    guidance_label_path: str = "data/crossdocked_vina_score_labels.jsonl"
    guidance_lower_is_better: bool = True
    guidance_loss: str = "mse"
    mask_positive_guidance_labels: bool = True

    # train params
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    precision: str = "bf16"
    target_loss_scale: float = 1.0

    project: str = "bindhard"
    entity: str = "rshulke-university-of-florida"
    run_name: str = "graphAttn_score_guidance"
    seed: int = 23


@dataclass
class OnlineRLConfig(TrainConfig):
    base_ckpt: str = "checkpoints/best.pt"
    updates: int = 1000

    rollout_batch_size: int = 4
    rollout_num_samples: int = 8
    num_inner_epochs: int = 1
    rollout_steps: int = 100
    rollout_time_schedule: str = "log"
    rollout_type_temperature: float = 0.01

    nft_beta: float = 0.3
    nft_beta_discrete: float = 0.3
    ema_decay: float = 0.995

    reward_work_dir: str = "tmp_vina_online_rl"
    reward_norm_ema_decay: float = 0.95
    reward_min_std: float = 1e-3
    reward_vina_clip_low: float = -16.0
    reward_vina_clip_high: float = -1.0
    reward_vina_offset: float = 1.0
    reward_vina_divisor: float = 15.0
    reward_sa_shift: float = 0.17
    reward_sa_scale: float = 0.83

    lr: float = 5e-6
    weight_decay: float = 0.0
    optimizer: str = "adam"
    grad_clip: float = 8.0
    log_every: int = 10
    run_name: str = "graphAttn_online_rl_vina_sa"
