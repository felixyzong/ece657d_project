import copy
from pathlib import Path

import torch
import torch.optim as optim

try:
    from .config import CFG, ensure_parent_dir, get_device, set_seed, setup_torch_cache
    from .kd_loss import kd_loss
    from .network import (
        build_class_conditional_resnet_generator,
        build_student_model,
        build_teacher_model,
        infer_num_classes_from_ckpt,
    )
    from .utils import normalize_for_teacher, sample_labels, sample_latent, unwrap_state_dict
except ImportError:
    from config import CFG, ensure_parent_dir, get_device, set_seed, setup_torch_cache
    from kd_loss import kd_loss
    from network import (
        build_class_conditional_resnet_generator,
        build_student_model,
        build_teacher_model,
        infer_num_classes_from_ckpt,
    )
    from utils import normalize_for_teacher, sample_labels, sample_latent, unwrap_state_dict


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


def load_generator(num_classes, device, ckpt_path):
    generator = build_class_conditional_resnet_generator(
        num_classes=num_classes,
        z_dim=CFG.z_dim,
        base_channels=CFG.gen_base_channels,
        image_size=CFG.image_size,
        device=device,
    )

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found: {path}. "
            "Run train_generator.py first."
        )

    state = torch.load(path, map_location=device)
    state_dict = unwrap_state_dict(state)
    generator.load_state_dict(state_dict, strict=True)

    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    return generator


def train_student_with_generator():
    set_seed(CFG.seed)
    setup_torch_cache(CFG.torch_cache_dir)
    device = get_device()

    num_classes = infer_num_classes_from_ckpt(CFG.teacher_ckpt_path, CFG.num_classes)
    teacher = load_teacher(num_classes=num_classes, device=device, ckpt_path=CFG.teacher_ckpt_path)
    generator = load_generator(num_classes=num_classes, device=device, ckpt_path=CFG.generator_ckpt_path)
    student = build_student_model(
        num_classes=num_classes,
        device=device,
        use_pretrained=CFG.student_pretrained_backbone,
    )

    optimizer = optim.Adam(
        student.parameters(),
        lr=CFG.student_lr,
        weight_decay=CFG.student_weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CFG.student_lr_step_size,
        gamma=CFG.student_lr_gamma,
    )

    best_kd_loss = float("inf")
    best_student_wts = copy.deepcopy(student.state_dict())

    print(
        f"Train student with generator samples | "
        f"steps={CFG.student_train_steps} | batch={CFG.student_batch_size} | num_classes={num_classes}"
    )

    for step in range(1, CFG.student_train_steps + 1):
        labels = sample_labels(CFG.student_batch_size, num_classes, device)
        z = sample_latent(CFG.student_batch_size, CFG.z_dim, device)

        with torch.no_grad():
            images_01 = generator(z, labels)
            images_for_teacher = normalize_for_teacher(images_01)
            teacher_logits = teacher(images_for_teacher)

        student.train()
        optimizer.zero_grad()

        student_logits = student(images_for_teacher)
        loss = kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            alpha=CFG.kd_alpha,
            temperature=CFG.kd_temperature,
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        preds = student_logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        if loss_val < best_kd_loss:
            best_kd_loss = loss_val
            best_student_wts = copy.deepcopy(student.state_dict())
            ensure_parent_dir(CFG.student_save_path)
            torch.save(best_student_wts, CFG.student_save_path)

        if step == 1 or step % CFG.student_log_interval == 0:
            print(
                f"Step [{step}/{CFG.student_train_steps}] | "
                f"KD Loss: {loss_val:.4f} | Synth Acc: {acc:.4f}"
            )

    student.load_state_dict(best_student_wts)
    ensure_parent_dir(CFG.student_save_path)
    torch.save(student.state_dict(), CFG.student_save_path)
    print(
        f"Saved best student to {CFG.student_save_path} | "
        f"Best KD Loss: {best_kd_loss:.4f}"
    )


if __name__ == "__main__":
    train_student_with_generator()
