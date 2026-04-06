import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch
from torchvision.utils import save_image

try:
    from .config import CFG, get_device, set_seed, setup_torch_cache
    from .deepinversion import DeepInversionSynthesizer
    from .network import infer_num_classes_from_ckpt
    from .train import load_teacher
except ImportError:
    from config import CFG, get_device, set_seed, setup_torch_cache
    from deepinversion import DeepInversionSynthesizer
    from network import infer_num_classes_from_ckpt
    from train import load_teacher


def _denormalize_imagenet(images: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def _save_synth_images(images: torch.Tensor, labels: torch.Tensor, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _denormalize_imagenet(images.detach().cpu())
    labels = labels.detach().cpu()

    saved_paths = []
    for idx in range(images.shape[0]):
        class_id = int(labels[idx].item())
        save_path = out_dir / f"{prefix}_idx_{idx:03d}_cls_{class_id}.png"
        save_image(images[idx], str(save_path))
        saved_paths.append(save_path)
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one DeepInversion round and save a batch of synthetic images."
    )
    parser.add_argument("--batch-size", type=int, default=CFG.student_batch_size)
    parser.add_argument("--inversion-steps", type=int, default=CFG.inversion_steps)
    parser.add_argument("--output-dir", type=str, default=CFG.save_synth_dir)
    parser.add_argument("--prefix", type=str, default="oneshot")
    return parser.parse_args()


def main():
    args = parse_args()

    runtime_cfg = replace(
        CFG,
        student_batch_size=args.batch_size,
        inversion_steps=args.inversion_steps,
        save_synth_dir=args.output_dir,
    )

    set_seed(runtime_cfg.seed)
    setup_torch_cache(runtime_cfg.torch_cache_dir)
    device = get_device()

    num_classes = infer_num_classes_from_ckpt(runtime_cfg.teacher_ckpt_path, runtime_cfg.num_classes)
    teacher = load_teacher(
        num_classes=num_classes,
        device=device,
        ckpt_path=runtime_cfg.teacher_ckpt_path,
    )

    inverter = DeepInversionSynthesizer(
        teacher=teacher,
        num_classes=num_classes,
        cfg=runtime_cfg,
        device=device,
    )

    print(
        f"Running one DeepInversion round | batch={runtime_cfg.student_batch_size} | "
        f"inversion_steps={runtime_cfg.inversion_steps}"
    )

    start = time.time()
    try:
        synth_images, synth_labels, inv_stats = inverter.synthesize(runtime_cfg.student_batch_size)
    finally:
        inverter.close()

    out_dir = Path(runtime_cfg.save_synth_dir)
    saved_paths = _save_synth_images(
        images=synth_images,
        labels=synth_labels,
        out_dir=out_dir,
        prefix=args.prefix,
    )

    elapsed = time.time() - start
    print(
        f"Done. Saved {len(saved_paths)} images to '{out_dir}'. "
        f"Inv CE={inv_stats['inv_loss_ce']:.4f}, Inv BN={inv_stats['inv_loss_r_feature']:.4f}, "
        f"Time={elapsed:.1f}s"
    )
    if saved_paths:
        print(f"Example: {saved_paths[0]}")


if __name__ == "__main__":
    main()
