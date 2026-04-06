from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20873157
    num_classes: int = 37
    image_size: int = 224

    teacher_ckpt_path: str = "models/best_resnet50_teacher.pth"
    save_path: str = "models/best_resnet18_student_distilled_00.pth"

    student_pretrained_backbone: bool = True
    teacher_pretrained_backbone: bool = True

    num_train_rounds: int = 200
    reuse_step: int = 5
    student_batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 100
    lr_gamma: float = 0.1

    kd_alpha: float = 0.5
    kd_temperature: float = 4.0

    inversion_steps: int = 1000
    inversion_lr: float = 0.1
    inversion_jitter: int = 30
    inversion_do_flip: bool = True

    di_ce_scale: float = 1.0
    di_r_feature: float = 0.01
    di_first_bn_multiplier: float = 10.0
    di_tv_l1: float = 1e-5
    di_tv_l2: float = 1e-4
    di_l2: float = 1e-4

    log_interval: int = 10
    save_synth_images: bool = True
    save_synth_every: int = 50
    save_synth_count: int = 8
    save_synth_dir: str = "data/synth"
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
