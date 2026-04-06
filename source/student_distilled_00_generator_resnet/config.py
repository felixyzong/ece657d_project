from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20873157
    num_classes: int = 37
    image_size: int = 224
    z_dim: int = 128
    gen_base_channels: int = 32

    teacher_ckpt_path: str = "models/best_resnet50_teacher.pth"
    generator_ckpt_path: str = "generator/class_conditional_resnet_generator.pth"
    student_save_path: str = "models/best_resnet18_student_distilled_00_generator_resnet.pth"

    teacher_pretrained_backbone: bool = True
    student_pretrained_backbone: bool = True

    gen_train_steps: int = 1500
    gen_batch_size: int = 16
    gen_lr: float = 2e-4
    gen_beta1: float = 0.5
    gen_beta2: float = 0.999
    gen_weight_decay: float = 0.0
    gen_log_interval: int = 20
    gen_save_image_every: int = 50
    gen_save_image_count: int = 16
    gen_save_image_dir: str = "data/synth_generator_resnet"

    gen_ce_scale: float = 1.0
    gen_r_feature: float = 0.01
    gen_first_bn_multiplier: float = 10.0
    gen_tv_l1: float = 0.0
    gen_tv_l2: float = 1e-5
    gen_l2: float = 1e-4

    student_train_steps: int = 1500
    student_batch_size: int = 32
    student_lr: float = 1e-3
    student_weight_decay: float = 1e-4
    student_lr_step_size: int = 500
    student_lr_gamma: float = 0.1
    student_log_interval: int = 20

    kd_alpha: float = 0.5
    kd_temperature: float = 4.0

    torch_cache_dir: str = ".torch_cache"


CFG = TrainConfig()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_torch_cache(cache_dir: str):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_path))


def ensure_parent_dir(file_path: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
