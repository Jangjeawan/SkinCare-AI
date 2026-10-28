"""분류 가중치를 한 스플릿에서 평가한다.

원본 ID는 Roboflow 파일명의 `.rf.` 앞부분이다. `--leak-root`를 주면
그 폴더(보통 train)와 ID가 겹치는 샘플을 빼고 잠금 수치를 따로 낸다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16

from train_classify import CLASSES, WoundDataset, build_transforms, list_samples


def origin_id(path: Path) -> str:
    stem = path.stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def ids_under(root: Path) -> set[str]:
    found: set[str] = set()
    for path in root.rglob("*.jpg"):
        if "unlabeled" in str(path).lower():
            continue
        found.add(origin_id(path))
    return found


def confusion(pred: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, guess in zip(targets.tolist(), pred.tolist()):
        matrix[truth][guess] += 1
    return matrix


def metrics_from_counts(matrix: list[list[int]]) -> dict:
    num_classes = len(matrix)
    per_class = []
    f1s = []
    for label in range(num_classes):
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(matrix[row][label] for row in range(num_classes))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
        per_class.append(
            {
                "label": CLASSES[label],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"macro_f1": sum(f1s) / num_classes, "per_class": per_class, "confusion": matrix}


@torch.no_grad()
def predict(model, rows, device, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        WoundDataset(rows, build_transforms(False)),
        batch_size=batch,
        shuffle=False,
        num_workers=0,
    )
    logits_all = []
    targets_all = []
    for images, targets in loader:
        logits_all.append(model(images.to(device)).cpu())
        targets_all.append(targets)
    return torch.cat(logits_all).argmax(dim=1), torch.cat(targets_all)


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--leak-root", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list_samples(args.data, args.split)
    model = load_model(args.checkpoint, device)
    pred, targets = predict(model, rows, device, args.batch)

    report: dict = {"checkpoint": str(args.checkpoint), "split": args.split, "n": len(rows)}
    report["all"] = metrics_from_counts(confusion(pred, targets, len(CLASSES)))

    if args.leak_root is not None:
        leaked_ids = ids_under(args.leak_root)
        keep = [index for index, (path, _) in enumerate(rows) if origin_id(path) not in leaked_ids]
        drop = [index for index in range(len(rows)) if index not in set(keep)]
        report["locked_n"] = len(keep)
        report["leaked_n"] = len(drop)
        if keep:
            index = torch.tensor(keep)
            report["locked"] = metrics_from_counts(confusion(pred[index], targets[index], len(CLASSES)))
        if drop:
            index = torch.tensor(drop)
            report["leaked_reference"] = metrics_from_counts(
                confusion(pred[index], targets[index], len(CLASSES))
            )

    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
