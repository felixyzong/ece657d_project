import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepInversionFeatureHook:
    """Collect BN feature regularization used in DeepInversion."""

    def __init__(self, module: nn.Module):
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


def image_prior_losses(images: torch.Tensor):
    diff1 = images[:, :, :, :-1] - images[:, :, :, 1:]
    diff2 = images[:, :, :-1, :] - images[:, :, 1:, :]
    diff3 = images[:, :, 1:, :-1] - images[:, :, :-1, 1:]
    diff4 = images[:, :, :-1, :-1] - images[:, :, 1:, 1:]

    loss_var_l2 = (
        diff1.pow(2).mean() + diff2.pow(2).mean() + diff3.pow(2).mean() + diff4.pow(2).mean()
    )
    loss_var_l1 = (
        diff1.abs().mean() + diff2.abs().mean() + diff3.abs().mean() + diff4.abs().mean()
    )
    return loss_var_l1, loss_var_l2


def clip_images_to_valid_range(images: torch.Tensor):
    device = images.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return torch.max(torch.min(images, upper), lower)


class DeepInversionSynthesizer:
    def __init__(self, teacher: nn.Module, num_classes: int, cfg, device):
        self.teacher = teacher
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device

        self.feature_hooks = [
            DeepInversionFeatureHook(m) for m in self.teacher.modules() if isinstance(m, nn.BatchNorm2d)
        ]

    def close(self):
        for hook in self.feature_hooks:
            hook.close()

    def _sample_targets(self, batch_size: int):
        return torch.randint(0, self.num_classes, (batch_size,), device=self.device)

    def synthesize(self, batch_size: int):
        self.teacher.eval()

        targets = self._sample_targets(batch_size)
        inputs = torch.randn(
            (batch_size, 3, self.cfg.image_size, self.cfg.image_size),
            device=self.device,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([inputs], lr=self.cfg.inversion_lr, betas=(0.5, 0.9))

        best_loss = float("inf")
        best_inputs = None
        best_stats = None

        for _ in range(self.cfg.inversion_steps):
            shifted = inputs
            jitter = int(self.cfg.inversion_jitter)
            if jitter > 0:
                off1 = int(torch.randint(-jitter, jitter + 1, (1,), device=self.device).item())
                off2 = int(torch.randint(-jitter, jitter + 1, (1,), device=self.device).item())
                shifted = torch.roll(shifted, shifts=(off1, off2), dims=(2, 3))

            if self.cfg.inversion_do_flip and torch.rand(1).item() > 0.5:
                shifted = torch.flip(shifted, dims=(3,))

            optimizer.zero_grad()
            self.teacher.zero_grad(set_to_none=True)

            teacher_logits = self.teacher(shifted)

            loss_ce = F.cross_entropy(teacher_logits, targets)
            loss_tv_l1, loss_tv_l2 = image_prior_losses(shifted)
            loss_l2 = shifted.pow(2).mean()

            if self.feature_hooks:
                bn_scales = [self.cfg.di_first_bn_multiplier] + [1.0] * (len(self.feature_hooks) - 1)
                loss_r_feature = sum(
                    hook.r_feature * bn_scales[idx] for idx, hook in enumerate(self.feature_hooks)
                )
            else:
                loss_r_feature = torch.tensor(0.0, device=self.device)

            loss = (
                self.cfg.di_ce_scale * loss_ce
                + self.cfg.di_r_feature * loss_r_feature
                + self.cfg.di_tv_l1 * loss_tv_l1
                + self.cfg.di_tv_l2 * loss_tv_l2
                + self.cfg.di_l2 * loss_l2
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                inputs.data = clip_images_to_valid_range(inputs.data)

            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_inputs = inputs.detach().clone()
                best_stats = {
                    "inv_loss_total": loss_val,
                    "inv_loss_ce": float(loss_ce.item()),
                    "inv_loss_r_feature": float(loss_r_feature.item()),
                }

        return best_inputs, targets.detach(), best_stats
