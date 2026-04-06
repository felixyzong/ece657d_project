from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

try:
    from .config import TrainConfig
except ImportError:
    from config import TrainConfig


class FlatFilenameDataset(Dataset):
    """Load images from a flat directory with file names: [Class_Name]_idx.jpg"""

    IMG_EXT = ".jpg"

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        image_paths = sorted(
            p for p in self.data_dir.iterdir()
            if p.is_file() and p.suffix.lower() == self.IMG_EXT
        )
        if not image_paths:
            raise RuntimeError(f"No image files found in: {self.data_dir}")

        class_name_by_path = {}
        class_set = set()
        for path in image_paths:
            stem = path.stem
            if "_" not in stem:
                raise ValueError(
                    f"Invalid filename '{path.name}'. Expected format: [Class_Name]_idx.jpg"
                )
            class_name = stem.rsplit("_", 1)[0]
            if not class_name:
                raise ValueError(
                    f"Invalid filename '{path.name}'. Class_Name cannot be empty."
                )
            class_name_by_path[path] = class_name
            class_set.add(class_name)

        self.classes = sorted(class_set)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = [
            (path, self.class_to_idx[class_name_by_path[path]])
            for path in image_paths
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_transform


def _build_split_indices(total_size: int, val_ratio: float, seed: int):
    if total_size < 2:
        raise ValueError("Dataset must contain at least 2 images for train/val split.")

    val_size = int(total_size * val_ratio)
    val_size = max(1, min(val_size, total_size - 1))

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def _take_fraction(indices, labels, fraction: float, seed: int):
    if fraction == 1.0:
        return indices

    target = max(1, int(len(indices) * fraction))
    generator = torch.Generator().manual_seed(seed)

    class_to_indices = {}
    for idx in indices:
        label = labels[idx]
        class_to_indices.setdefault(label, []).append(idx)

    classes = sorted(class_to_indices.keys())

    for label in classes:
        items = class_to_indices[label]
        order = torch.randperm(len(items), generator=generator).tolist()
        class_to_indices[label] = [items[i] for i in order]

    class_order = torch.randperm(len(classes), generator=generator).tolist()
    classes = [classes[i] for i in class_order]

    selected = []
    if target >= len(classes):
        for label in classes:
            selected.append(class_to_indices[label].pop())
    else:
        for label in classes[:target]:
            selected.append(class_to_indices[label].pop())
        return selected

    remaining = target - len(selected)
    if remaining <= 0:
        return selected

    pool = []
    for label in classes:
        pool.extend(class_to_indices[label])

    if not pool:
        return selected

    order = torch.randperm(len(pool), generator=generator).tolist()
    selected.extend(pool[i] for i in order[:remaining])
    return selected


def build_dataloaders(cfg: TrainConfig):
    train_transform, val_transform = build_transforms(cfg.image_size)

    train_base = FlatFilenameDataset(cfg.data_dir, transform=train_transform)
    val_base = FlatFilenameDataset(cfg.data_dir, transform=val_transform)

    train_indices, val_indices = _build_split_indices(
        total_size=len(train_base),
        val_ratio=cfg.val_ratio,
        seed=cfg.seed
    )
    sample_labels = [label for _, label in train_base.samples]
    train_indices = _take_fraction(
        train_indices,
        sample_labels,
        cfg.train_fraction,
        cfg.seed
    )

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    class_names = train_base.classes
    return train_loader, val_loader, class_names
