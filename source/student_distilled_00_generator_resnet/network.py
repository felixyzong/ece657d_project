from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        nn.init.zeros_(self.embed.weight)
        with torch.no_grad():
            self.embed.weight[:, :num_features].fill_(1.0)

        self.num_features = num_features

    def forward(self, x, labels):
        out = self.bn(x)
        gamma_beta = self.embed(labels)
        gamma, beta = torch.split(gamma_beta, self.num_features, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class ResBlockUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        h = self.cbn1(x, labels)
        h = F.relu(h, inplace=True)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.cbn2(h, labels)
        h = F.relu(h, inplace=True)
        h = self.conv2(h)

        x_sc = F.interpolate(x, scale_factor=2, mode="nearest")
        x_sc = self.conv_sc(x_sc)
        return h + x_sc


class SpatialSelfAttention(nn.Module):
    """Apply MHA on spatial tokens of a feature map."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).permute(0, 2, 1).contiguous()  # [B, HW, C]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attn_out)
        out = tokens.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return out


class ClassConditionalResNetGenerator(nn.Module):
    def __init__(self, num_classes: int, z_dim: int = 128, base_channels: int = 64, image_size: int = 224):
        super().__init__()
        self.z_dim = z_dim
        self.image_size = image_size

        self.class_embed = nn.Embedding(num_classes, z_dim)
        self.linear = nn.Linear(z_dim, base_channels * 16 * 7 * 7)

        self.block1 = ResBlockUp(base_channels * 16, base_channels * 16, num_classes)  # 7 -> 14
        self.block2 = ResBlockUp(base_channels * 16, base_channels * 8, num_classes)   # 14 -> 28
        self.block3 = ResBlockUp(base_channels * 8, base_channels * 4, num_classes)     # 28 -> 56
        self.attn = SpatialSelfAttention(base_channels * 4, num_heads=4)
        self.block4 = ResBlockUp(base_channels * 4, base_channels * 2, num_classes)     # 56 -> 112
        self.block5 = ResBlockUp(base_channels * 2, base_channels, num_classes)          # 112 -> 224

        self.cbn_final = ConditionalBatchNorm2d(base_channels, num_classes)
        self.conv_final = nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z, labels):
        z = z + self.class_embed(labels)

        h = self.linear(z)
        h = h.view(h.size(0), -1, 7, 7)

        h = self.block1(h, labels)
        h = self.block2(h, labels)
        h = self.block3(h, labels)
        h = self.attn(h)
        h = self.block4(h, labels)
        h = self.block5(h, labels)

        h = self.cbn_final(h, labels)
        h = F.relu(h, inplace=True)
        h = self.conv_final(h)
        x = torch.sigmoid(h)

        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x


def build_class_conditional_resnet_generator(num_classes: int, z_dim: int, base_channels: int, image_size: int, device):
    model = ClassConditionalResNetGenerator(
        num_classes=num_classes,
        z_dim=z_dim,
        base_channels=base_channels,
        image_size=image_size,
    )
    return model.to(device)


def build_student_model(num_classes: int, device, use_pretrained: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def build_teacher_model(num_classes: int, device, use_pretrained: bool = False):
    weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def infer_num_classes_from_ckpt(ckpt_path: str, fallback_num_classes: int):
    path = Path(ckpt_path)
    if not path.exists():
        return fallback_num_classes

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state)

    if isinstance(state, dict):
        for key in ("fc.weight", "module.fc.weight"):
            if key in state:
                return int(state[key].shape[0])

    return fallback_num_classes
