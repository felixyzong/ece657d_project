from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/oxford-iiit-pet/images"
    batch_size: int = 32
    val_ratio: float = 0.2
    train_fraction: float = 1.0
    num_workers: int = 0
    seed: int = 20873157
    image_size: int = 224
    models_dir: str = "models"
    report_path: str = "source/evaluate/eval_report.txt"
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
