import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from .config import CFG, ensure_parent_dir, get_device, set_seed, setup_torch_cache
    from .network import (
        build_class_conditional_resnet_generator,
        build_teacher_model,
        infer_num_classes_from_ckpt,
    )
    from .utils import (
        close_hooks,
        compute_bn_feature_loss,
        image_prior_losses,
        normalize_for_teacher,
        register_bn_feature_hooks,
        sample_labels,
        sample_latent,
        save_synthetic_images,
        unwrap_state_dict,
    )
except ImportError:
    from config import CFG, ensure_parent_dir, get_device, set_seed, setup_torch_cache
    from network import (
        build_class_conditional_resnet_generator,
        build_teacher_model,
        infer_num_classes_from_ckpt,
    )
    from utils import (
        close_hooks,
        compute_bn_feature_loss,
        image_prior_losses,
        normalize_for_teacher,
        register_bn_feature_hooks,
        sample_labels,
        sample_latent,
        save_synthetic_images,
        unwrap_state_dict,
    )


def _save_generator_checkpoint(generator_state_dict, num_classes):
    ensure_parent_dir(CFG.generator_ckpt_path)
    torch.save(
        {
            "state_dict": generator_state_dict,
            "num_classes": num_classes,
            "z_dim": CFG.z_dim,
            "gen_base_channels": CFG.gen_base_channels,
            "image_size": CFG.image_size,
        },
        CFG.generator_ckpt_path,
    )


def load_teacher(num_classes, device, ckpt_path):
    teacher = build_teacher_model(
        num_classes=num_classes,
        device=device,
        use_pretrained=CFG.teacher_pretrained_backbone,
    )

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}. "
            "Train teacher first or update CFG.teacher_ckpt_path."
        )

    state = torch.load(path, map_location=device)
    state_dict = unwrap_state_dict(state)
    teacher.load_state_dict(state_dict, strict=True)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def train_generator():
    set_seed(CFG.seed)
    setup_torch_cache(CFG.torch_cache_dir)
    device = get_device()

    num_classes = infer_num_classes_from_ckpt(CFG.teacher_ckpt_path, CFG.num_classes)
    teacher = load_teacher(num_classes=num_classes, device=device, ckpt_path=CFG.teacher_ckpt_path)
    generator = build_class_conditional_resnet_generator(
        num_classes=num_classes,
        z_dim=CFG.z_dim,
        base_channels=CFG.gen_base_channels,
        image_size=CFG.image_size,
        device=device,
    )

    optimizer = optim.Adam(
        generator.parameters(),
        lr=CFG.gen_lr,
        betas=(CFG.gen_beta1, CFG.gen_beta2),
        weight_decay=CFG.gen_weight_decay,
    )

    hooks = register_bn_feature_hooks(teacher)
    fixed_labels = torch.arange(CFG.gen_save_image_count, device=device) % num_classes
    fixed_z = sample_latent(CFG.gen_save_image_count, CFG.z_dim, device)

    best_loss = float("inf")
    best_generator_wts = copy.deepcopy(generator.state_dict())

    print(
        f"Train class-conditional ResNet generator | "
        f"steps={CFG.gen_train_steps} | batch={CFG.gen_batch_size} | num_classes={num_classes}"
    )

    try:
        for step in range(1, CFG.gen_train_steps + 1):
            start_t = time.time()

            labels = sample_labels(CFG.gen_batch_size, num_classes, device)
            z = sample_latent(CFG.gen_batch_size, CFG.z_dim, device)

            generator.train()
            optimizer.zero_grad()

            images_01 = generator(z, labels)
            images_for_teacher = normalize_for_teacher(images_01)
            teacher_logits = teacher(images_for_teacher)

            loss_ce = F.cross_entropy(teacher_logits, labels)
            loss_r_feature = compute_bn_feature_loss(hooks, CFG.gen_first_bn_multiplier)
            if not torch.is_tensor(loss_r_feature):
                loss_r_feature = torch.tensor(float(loss_r_feature), device=device)
            else:
                loss_r_feature = loss_r_feature.to(device)

            loss_tv_l1, loss_tv_l2 = image_prior_losses(images_01)
            loss_l2 = images_01.pow(2).mean()

            loss = (
                CFG.gen_ce_scale * loss_ce
                + CFG.gen_r_feature * loss_r_feature
                + CFG.gen_tv_l1 * loss_tv_l1
                + CFG.gen_tv_l2 * loss_tv_l2
                + CFG.gen_l2 * loss_l2
            )

            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_generator_wts = copy.deepcopy(generator.state_dict())
                _save_generator_checkpoint(best_generator_wts, num_classes)

            if step == 1 or step % CFG.gen_log_interval == 0:
                elapsed = time.time() - start_t
                print(
                    f"Step [{step}/{CFG.gen_train_steps}] | "
                    f"G Loss: {loss_val:.4f} | CE: {loss_ce.item():.4f} | "
                    f"BN: {loss_r_feature.item():.4f} | Time: {elapsed:.1f}s"
                )

            if step % CFG.gen_save_image_every == 0:
                generator.eval()
                with torch.no_grad():
                    sample_images = generator(fixed_z, fixed_labels)
                save_synthetic_images(
                    images_01=sample_images,
                    labels=fixed_labels,
                    output_dir=CFG.gen_save_image_dir,
                    prefix=f"step_{step:05d}",
                    max_count=CFG.gen_save_image_count,
                )
    finally:
        close_hooks(hooks)

    generator.load_state_dict(best_generator_wts)
    _save_generator_checkpoint(generator.state_dict(), num_classes)

    generator.eval()
    with torch.no_grad():
        final_images = generator(fixed_z, fixed_labels)
    save_synthetic_images(
        images_01=final_images,
        labels=fixed_labels,
        output_dir=CFG.gen_save_image_dir,
        prefix="final",
        max_count=CFG.gen_save_image_count,
    )

    print(
        f"Saved generator checkpoint to {CFG.generator_ckpt_path} | "
        f"Best Loss: {best_loss:.4f} | Sample images: {CFG.gen_save_image_dir}"
    )


if __name__ == "__main__":
    train_generator()
