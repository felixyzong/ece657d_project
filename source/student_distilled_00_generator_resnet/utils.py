from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_for_teacher(images_01: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=images_01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images_01.device).view(1, 3, 1, 1)
    return (images_01 - mean) / std


def image_prior_losses(images_01: torch.Tensor):
    diff1 = images_01[:, :, :, :-1] - images_01[:, :, :, 1:]
    diff2 = images_01[:, :, :-1, :] - images_01[:, :, 1:, :]
    diff3 = images_01[:, :, 1:, :-1] - images_01[:, :, :-1, 1:]
    diff4 = images_01[:, :, :-1, :-1] - images_01[:, :, 1:, 1:]

    loss_var_l2 = (
        diff1.pow(2).mean() + diff2.pow(2).mean() + diff3.pow(2).mean() + diff4.pow(2).mean()
    )
    loss_var_l1 = (
        diff1.abs().mean() + diff2.abs().mean() + diff3.abs().mean() + diff4.abs().mean()
    )
    return loss_var_l1, loss_var_l2


def save_synthetic_images(images_01, labels, output_dir, prefix, max_count):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = min(max_count, images_01.shape[0])
    images = images_01[:count].detach().cpu().clamp(0.0, 1.0)
    labels = labels[:count].detach().cpu()

    saved = []
    for idx in range(count):
        class_id = int(labels[idx].item())
        save_path = out_dir / f"{prefix}_idx_{idx:03d}_cls_{class_id}.png"
        save_image(images[idx], str(save_path))
        saved.append(save_path)
    return saved


def sample_latent(batch_size: int, z_dim: int, device):
    return torch.randn(batch_size, z_dim, device=device)


def sample_labels(batch_size: int, num_classes: int, device):
    return torch.randint(0, num_classes, (batch_size,), device=device)


class BNFeatureHook:
    def __init__(self, module: nn.BatchNorm2d):
        self.r_feature = torch.tensor(0.0, device=module.running_mean.device)
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, module_in, module_out):
        del module_out
        x = module_in[0]
        nch = x.shape[1]

        mean = x.mean(dim=[0, 2, 3])
        var = x.permute(1, 0, 2, 3).contiguous().view(nch, -1).var(dim=1, unbiased=False)

        self.r_feature = torch.norm(module.running_var - var, p=2) + torch.norm(
            module.running_mean - mean, p=2
        )

    def close(self):
        self.hook.remove()


def register_bn_feature_hooks(model):
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(BNFeatureHook(m))
    return hooks


def compute_bn_feature_loss(hooks, first_bn_multiplier: float):
    if not hooks:
        return torch.tensor(0.0)

    scales = [first_bn_multiplier] + [1.0] * (len(hooks) - 1)
    return sum(hook.r_feature * scales[idx] for idx, hook in enumerate(hooks))


def close_hooks(hooks):
    for hook in hooks:
        hook.close()


def unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        inner = checkpoint_obj.get("state_dict")
        if isinstance(inner, dict):
            checkpoint_obj = inner

    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Checkpoint is not a valid state_dict dictionary.")

    cleaned = {}
    for key, value in checkpoint_obj.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned
