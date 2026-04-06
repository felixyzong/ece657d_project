import copy
import time
from pathlib import Path

import torch
import torch.optim as optim
from torchvision.utils import save_image

try:
    from .config import CFG, ensure_parent_dir, get_device, set_seed, setup_torch_cache
    from .deepinversion import DeepInversionSynthesizer
    from .kd_loss import kd_loss
    from .network import (
        build_student_distilled_model,
        build_teacher_model,
        infer_num_classes_from_ckpt,
    )
except ImportError:
    from config import CFG, ensure_parent_dir, get_device, set_seed, setup_torch_cache
    from deepinversion import DeepInversionSynthesizer
    from kd_loss import kd_loss
    from network import (
        build_student_distilled_model,
        build_teacher_model,
        infer_num_classes_from_ckpt,
    )


def load_teacher(num_classes, device, ckpt_path):
    teacher = build_teacher_model(
        num_classes=num_classes,
        device=device,
    )

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}. "
            "Train teacher first or update CFG.teacher_ckpt_path."
        )

    state = torch.load(path, map_location=device)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    teacher.load_state_dict(state_dict, strict=True)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _denormalize_imagenet(images: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def maybe_save_synth_images(images, labels, round_idx):
    if not CFG.save_synth_images:
        return
    if CFG.save_synth_every <= 0 or round_idx % CFG.save_synth_every != 0:
        return

    out_dir = Path(CFG.save_synth_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = min(CFG.save_synth_count, images.shape[0])
    images_to_save = _denormalize_imagenet(images[:count].detach().cpu())
    labels_to_save = labels[:count].detach().cpu()

    for idx in range(count):
        class_id = int(labels_to_save[idx].item())
        save_path = out_dir / f"round_{round_idx:05d}_idx_{idx:02d}_cls_{class_id}.png"
        save_image(images_to_save[idx], str(save_path))


def train_data_free():
    set_seed(CFG.seed)
    setup_torch_cache(CFG.torch_cache_dir)
    device = get_device()

    num_classes = infer_num_classes_from_ckpt(CFG.teacher_ckpt_path, CFG.num_classes)

    teacher = load_teacher(
        num_classes=num_classes,
        device=device,
        ckpt_path=CFG.teacher_ckpt_path
    )
    student = build_student_distilled_model(
        num_classes=num_classes,
        device=device
    )

    optimizer = optim.Adam(
        student.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CFG.lr_step_size,
        gamma=CFG.lr_gamma,
    )

    inverter = DeepInversionSynthesizer(
        teacher=teacher,
        num_classes=num_classes,
        cfg=CFG,
        device=device,
    )

    best_loss = float("inf")
    best_student_wts = copy.deepcopy(student.state_dict())

    print(
        f"Data-free KD with DeepInversion | num_classes={num_classes} | "
        f"train_rounds={CFG.num_train_rounds} | reuse_step={CFG.reuse_step} | "
        f"total_updates={CFG.num_train_rounds * CFG.reuse_step} | "
        f"synth_batch={CFG.student_batch_size}"
    )

    try:
        global_update = 0
        for round_idx in range(1, CFG.num_train_rounds + 1):
            round_start_t = time.time()

            synth_images, synth_labels, inv_stats = inverter.synthesize(CFG.student_batch_size)
            with torch.no_grad():
                teacher_logits = teacher(synth_images)

            maybe_save_synth_images(synth_images, synth_labels, round_idx)

            round_loss_sum = 0.0
            round_acc_sum = 0.0

            for _ in range(CFG.reuse_step):
                student.train()
                optimizer.zero_grad()

                student_logits = student(synth_images)
                loss = kd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=synth_labels,
                    alpha=CFG.kd_alpha,
                    temperature=CFG.kd_temperature,
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                preds = student_logits.argmax(dim=1)
                acc = (preds == synth_labels).float().mean().item()
                loss_val = float(loss.item())

                round_loss_sum += loss_val
                round_acc_sum += acc
                global_update += 1

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_student_wts = copy.deepcopy(student.state_dict())
                    ensure_parent_dir(CFG.save_path)
                    torch.save(best_student_wts, CFG.save_path)

            if round_idx == 1 or round_idx % CFG.log_interval == 0:
                elapsed = time.time() - round_start_t
                avg_round_loss = round_loss_sum / CFG.reuse_step
                avg_round_acc = round_acc_sum / CFG.reuse_step
                print(
                    f"Round [{round_idx}/{CFG.num_train_rounds}] | "
                    f"Update [{global_update}/{CFG.num_train_rounds * CFG.reuse_step}] | "
                    f"KD Loss(avg): {avg_round_loss:.4f} | Synth Acc(avg): {avg_round_acc:.4f} | "
                    f"Inv CE: {inv_stats['inv_loss_ce']:.4f} | "
                    f"Inv BN: {inv_stats['inv_loss_r_feature']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
    finally:
        inverter.close()

    student.load_state_dict(best_student_wts)
    ensure_parent_dir(CFG.save_path)
    torch.save(student.state_dict(), CFG.save_path)
    print(f"Saved best student to {CFG.save_path} | Best KD Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_data_free()
