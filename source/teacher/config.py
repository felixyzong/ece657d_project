from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/oxford-iiit-pet/images"
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    num_workers: int = 4
    seed: int = 20873157
    image_size: int = 224
    lr_step_size: int = 5
    lr_gamma: float = 0.1
    save_path: str = "models/best_resnet50_teacher.pth"
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
