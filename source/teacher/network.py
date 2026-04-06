import torch.nn as nn
from torchvision import models


def build_teacher_model(num_classes: int, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
