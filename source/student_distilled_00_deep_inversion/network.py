from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_student_distilled_model(num_classes: int, device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def build_teacher_model(num_classes: int, device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def infer_num_classes_from_ckpt(ckpt_path: str, fallback_num_classes: int):
    path = Path(ckpt_path)
    if not path.exists():
        return fallback_num_classes

    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state

    for key in ("fc.weight", "module.fc.weight"):
        if isinstance(state_dict, dict) and key in state_dict:
            return int(state_dict[key].shape[0])

    return fallback_num_classes
