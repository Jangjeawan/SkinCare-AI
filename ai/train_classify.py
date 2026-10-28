"""유형 분류 학습. 폴더명 laceration 은 open_wound 로 넣고 Unlabeled 는 빼 둔다."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

CLASSES = ["abrasion", "bruise", "burn", "open_wound", "normal_skin"]
FOLDER_TO_CLASS = {
    "abrasion": "abrasion",
    "bruise": "bruise",
    "burn": "burn",
    "laceration": "open_wound",
    "normal skin": "normal_skin",
}


def list_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    split_dir = root / split
    for folder, name in FOLDER_TO_CLASS.items():
        label = CLASSES.index(name)
        folder_path = split_dir / folder
        if not folder_path.is_dir():
            continue
        for path in folder_path.glob("*.jpg"):
            rows.append((path, label))
    if not rows:
        raise SystemExit(f"이미지가 없습니다: {split_dir}")
    return rows


class WoundDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, label = self.rows[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    shared = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
                *shared,
            ]
        )
    return transforms.Compose([transforms.Resize((224, 224)), *shared])


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    f1s = []
    for label in range(num_classes):
        tp = ((pred == label) & (targets == label)).sum().item()
        fp = ((pred == label) & (targets != label)).sum().item()
        fn = ((pred != label) & (targets == label)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(f1s) / num_classes


def class_weights(rows: list[tuple[Path, int]], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for _, label in rows:
        counts[label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    logits_all = []
    targets_all = []
    for images, targets in loader:
        images = images.to(device)
        logits_all.append(model(images).cpu())
        targets_all.append(targets)
    return macro_f1(torch.cat(logits_all), torch.cat(targets_all), len(CLASSES))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/classify"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_rows = list_samples(args.data, "train")
    valid_rows = list_samples(args.data, "valid")
    print(f"train={len(train_rows)} valid={len(valid_rows)}", flush=True)

    train_set = WoundDataset(train_rows, build_transforms(True))
    valid_set = WoundDataset(valid_rows, build_transforms(False))
    sample_weights = [class_weights(train_rows, len(CLASSES))[label].item() for _, label in train_rows]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False, num_workers=0)

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows, len(CLASSES)).to(device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    head_params = list(model.heads.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()
        val_f1 = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": running / seen, "valid_macro_f1": val_f1}
        history.append(row)
        print(f"epoch {epoch} loss {row['train_loss']:.4f} valid_macro_f1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            stale = 0
            torch.save(
                {"model": model.state_dict(), "classes": CLASSES, "valid_macro_f1": best_f1},
                args.out / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_valid_macro_f1 {best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
