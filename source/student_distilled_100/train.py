import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import CFG, get_device, set_seed, setup_torch_cache
    from .dataset import build_dataloaders
    from .network import build_student_distilled_model, build_teacher_model
except ImportError:
    from config import CFG, get_device, set_seed, setup_torch_cache
    from dataset import build_dataloaders
    from network import build_student_distilled_model, build_teacher_model

try:
    from .kd_loss import kd_loss
except ImportError:
    from kd_loss import kd_loss


def train_one_epoch(student, teacher, loader, optimizer, device, alpha, temperature):
    student.train()
    teacher.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        loss = kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            alpha=alpha,
            temperature=temperature
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = student_logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def load_teacher(num_classes, device, ckpt_path):
    teacher = build_teacher_model(num_classes=num_classes, device=device)

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}. "
            "Train teacher first or update CFG.teacher_ckpt_path."
        )

    state_dict = torch.load(path, map_location=device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    return teacher


def main():
    set_seed(CFG.seed)
    setup_torch_cache(CFG.torch_cache_dir)
    device = get_device()

    train_loader, val_loader, class_names = build_dataloaders(CFG)
    num_classes = len(class_names)

    print(
        f"Found {len(train_loader.dataset) + len(val_loader.dataset)} images across {num_classes} classes. "
        f"Using {len(train_loader.dataset)} samples for training ({CFG.train_fraction * 100:.0f}% of train split)."
    )

    student = build_student_distilled_model(num_classes=num_classes, device=device)
    teacher = load_teacher(
        num_classes=num_classes,
        device=device,
        ckpt_path=CFG.teacher_ckpt_path
    )

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        student.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CFG.lr_step_size,
        gamma=CFG.lr_gamma
    )

    best_val_acc = 0.0
    best_student_wts = copy.deepcopy(student.state_dict())

    for epoch in range(CFG.num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=CFG.kd_alpha,
            temperature=CFG.kd_temperature
        )
        val_loss, val_acc = validate(
            model=student,
            loader=val_loader,
            criterion=ce_criterion,
            device=device
        )

        scheduler.step()
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{CFG.num_epochs}] | "
            f"train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_student_wts = copy.deepcopy(student.state_dict())
            torch.save(best_student_wts, CFG.save_path)
            print(f"Saved best student model to {CFG.save_path}")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    student.load_state_dict(best_student_wts)


if __name__ == "__main__":
    main()
