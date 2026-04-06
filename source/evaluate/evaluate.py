from datetime import datetime
from pathlib import Path

import torch

try:
    from .config import CFG, get_device, set_seed, setup_torch_cache
    from .dataset import build_dataloaders
    from .network import build_resnet18, build_resnet50
except ImportError:
    from config import CFG, get_device, set_seed, setup_torch_cache
    from dataset import build_dataloaders
    from network import build_resnet18, build_resnet50


def _unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise TypeError("Checkpoint is not a valid state_dict dictionary.")


def _strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


def _infer_architecture(state_dict):
    fc_weight = state_dict.get("fc.weight")
    if fc_weight is None:
        raise KeyError("fc.weight not found in checkpoint.")

    in_features = int(fc_weight.shape[1])
    if in_features == 512:
        return "resnet18"
    if in_features == 2048:
        return "resnet50"
    raise ValueError(f"Unsupported architecture by fc.in_features={in_features}")


def _build_model(arch, num_classes, device):
    if arch == "resnet18":
        return build_resnet18(num_classes=num_classes, device=device)
    if arch == "resnet50":
        return build_resnet50(num_classes=num_classes, device=device)
    raise ValueError(f"Unsupported architecture: {arch}")


@torch.no_grad()
def _evaluate_split(model, loader, device, num_classes):
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        batch_size = labels.size(0)
        total += batch_size

        max_k = min(5, num_classes)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        top1 = pred[:, 0]

        top1_correct += (top1 == labels).sum().item()
        top5_correct += pred.eq(labels.view(-1, 1)).any(dim=1).sum().item()

        class_total += torch.bincount(labels.cpu(), minlength=num_classes)
        correct_mask = (top1 == labels)
        if correct_mask.any():
            class_correct += torch.bincount(labels[correct_mask].cpu(), minlength=num_classes)

    top1_acc = (top1_correct / total) * 100.0 if total > 0 else 0.0
    top5_acc = (top5_correct / total) * 100.0 if total > 0 else 0.0

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "class_correct": class_correct.tolist(),
        "class_total": class_total.tolist(),
        "total": total,
    }


def _format_per_class_lines(class_names, class_correct, class_total):
    lines = []
    for idx, class_name in enumerate(class_names):
        total = class_total[idx]
        correct = class_correct[idx]
        if total == 0:
            acc = 0.0
        else:
            acc = (correct / total) * 100.0
        lines.append(f"  - {class_name}: {acc:.2f}% ({correct}/{total})")
    return lines


def _evaluate_one_model(model_path, train_loader, val_loader, class_names, device):
    checkpoint = torch.load(model_path, map_location=device)
    raw_state_dict = _unwrap_state_dict(checkpoint)
    state_dict = _strip_module_prefix(raw_state_dict)

    arch = _infer_architecture(state_dict)
    model = _build_model(arch=arch, num_classes=len(class_names), device=device)
    model.load_state_dict(state_dict, strict=True)

    train_metrics = _evaluate_split(
        model=model,
        loader=train_loader,
        device=device,
        num_classes=len(class_names),
    )
    test_metrics = _evaluate_split(
        model=model,
        loader=val_loader,
        device=device,
        num_classes=len(class_names),
    )

    return {
        "model_name": model_path.name,
        "arch": arch,
        "train_top1": train_metrics["top1"],
        "train_top5": train_metrics["top5"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "train_test_gap": train_metrics["top1"] - test_metrics["top1"],
        "per_class_lines": _format_per_class_lines(
            class_names,
            test_metrics["class_correct"],
            test_metrics["class_total"],
        ),
        "status": "ok",
    }


def _generate_report(results, class_names, train_size, test_size):
    lines = []
    lines.append("Model Evaluation Report")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Dataset Split Setup (from source/evaluate config + dataset):")
    lines.append(f"- seed: {CFG.seed}")
    lines.append(f"- val_ratio: {CFG.val_ratio}")
    lines.append(f"- train_fraction: {CFG.train_fraction}")
    lines.append(f"- num_classes: {len(class_names)}")
    lines.append(f"- train_size: {train_size}")
    lines.append(f"- test_size(val split): {test_size}")
    lines.append("")

    for item in results:
        lines.append("=" * 80)
        lines.append(f"Model: {item['model_name']}")
        if item["status"] != "ok":
            lines.append(f"Status: failed")
            lines.append(f"Error: {item['error']}")
            lines.append("")
            continue

        lines.append(f"Architecture: {item['arch']}")
        lines.append(f"Top-1 Accuracy (train): {item['train_top1']:.2f}%")
        lines.append(f"Top-5 Accuracy (train): {item['train_top5']:.2f}%")
        lines.append(f"Top-1 Accuracy (test): {item['test_top1']:.2f}%")
        lines.append(f"Top-5 Accuracy (test): {item['test_top5']:.2f}%")
        lines.append(f"Train-Test Gap (top-1): {item['train_test_gap']:.2f}%")
        lines.append("Per-Class Accuracy (test):")
        lines.extend(item["per_class_lines"])
        lines.append("")

    return "\n".join(lines)


def main():
    set_seed(CFG.seed)
    setup_torch_cache(CFG.torch_cache_dir)
    device = get_device()

    train_loader, val_loader, class_names = build_dataloaders(CFG)

    model_paths = sorted(Path(CFG.models_dir).glob("*.pth"))
    if not model_paths:
        raise FileNotFoundError(f"No .pth model files found under: {CFG.models_dir}")

    print(
        f"Evaluating {len(model_paths)} model(s) using fixed split "
        f"(seed={CFG.seed}, val_ratio={CFG.val_ratio}, train_fraction={CFG.train_fraction})."
    )

    results = []
    for model_path in model_paths:
        print(f"- Evaluating {model_path.name} ...")
        try:
            result = _evaluate_one_model(
                model_path=model_path,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
            )
            results.append(result)
        except Exception as exc:
            results.append({
                "model_name": model_path.name,
                "status": "failed",
                "error": str(exc),
            })

    report_text = _generate_report(
        results=results,
        class_names=class_names,
        train_size=len(train_loader.dataset),
        test_size=len(val_loader.dataset),
    )

    report_path = Path(CFG.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
