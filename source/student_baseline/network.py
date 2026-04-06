import torch.nn as nn
from torchvision import models


def build_student_baseline_model(num_classes: int, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
