import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import CFG, get_device, set_seed, setup_torch_cache
    from .dataset import build_dataloaders
    from .network import build_student_baseline_model
except ImportError:
    from config import CFG, get_device, set_seed, setup_torch_cache
    from dataset import build_dataloaders
    from network import build_student_baseline_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
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

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main():
    set_seed(CFG.seed)
    setup_torch_cache(CFG.torch_cache_dir)
    device = get_device()

    train_loader, val_loader, class_names = build_dataloaders(CFG)
    num_classes = len(class_names)
    print(f"Found {len(train_loader.dataset) + len(val_loader.dataset)} images across {num_classes} classes.")

    model = build_student_baseline_model(num_classes=num_classes, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CFG.lr_step_size,
        gamma=CFG.lr_gamma
    )

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(CFG.num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{CFG.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, CFG.save_path)
            print(f"Saved best model to {CFG.save_path}")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    main()
