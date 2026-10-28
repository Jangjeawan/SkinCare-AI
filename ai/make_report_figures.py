"""설계서 9.2에 넣을 학습 성능 그림을 만든다.

분류 ROC는 잠금 test의 게이트 적용 전 확률로 그린다.
탐지 PR은 wound-detect test, IoU 0.5 기준이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from eval_classify import load_model
from train_classify import CLASSES, WoundDataset, build_transforms, list_samples

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "figures"
CLASSIFY_DATA = ROOT.parent / "datasets" / "classify_locked"
CLASSIFY_CKPT = ROOT / "runs" / "classify_locked" / "best.pt"
HISTORY = ROOT / "runs" / "classify_locked" / "history.json"
METRICS = ROOT / "runs" / "classify_locked" / "test_metrics.json"
DETECT_CKPT = ROOT / "runs" / "det_v8s_wound" / "weights" / "best.pt"
DETECT_YAML = ROOT / "wound-detect.yaml"

DISPLAY = {
    "abrasion": "찰과상",
    "bruise": "타박상",
    "burn": "화상",
    "open_wound": "벌어진 상처",
    "normal_skin": "정상 피부",
}
COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]


def setup_font() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def binary_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = labels[order].astype(float)
    positive = y.sum()
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    tpr = np.concatenate([[0.0], tps / positive, [1.0]])
    fpr = np.concatenate([[0.0], fps / negative, [1.0]])
    area = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, area


@torch.no_grad()
def classification_probabilities() -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list_samples(CLASSIFY_DATA, "test")
    model = load_model(CLASSIFY_CKPT, device)
    loader = DataLoader(
        WoundDataset(rows, build_transforms(False)),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    logits, targets = [], []
    for images, target in loader:
        logits.append(model(images.to(device)).cpu())
        targets.append(target)
    probs = torch.softmax(torch.cat(logits), dim=1).numpy()
    return probs, torch.cat(targets).numpy()


def plot_roc(probs: np.ndarray, targets: np.ndarray) -> dict:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    grid = np.linspace(0, 1, 201)
    mean_tpr = np.zeros_like(grid)
    aucs = {}
    for index, name in enumerate(CLASSES):
        fpr, tpr, area = binary_roc((targets == index).astype(int), probs[:, index])
        aucs[name] = round(area, 4)
        mean_tpr += np.interp(grid, fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[index], lw=1.8, label=f"{DISPLAY[name]}  {area:.3f}")
    mean_tpr /= len(CLASSES)
    macro = float(np.mean(list(aucs.values())))
    ax.plot(grid, mean_tpr, color="black", lw=2.2, linestyle="--", label=f"macro  {macro:.3f}")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("위양성률 (FPR)")
    ax.set_ylabel("참양성률 (TPR)")
    ax.set_title("분류 One-vs-Rest ROC  (잠금 test)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_9_2_1_roc.png", dpi=160)
    plt.close(fig)
    return {"per_class_auc": aucs, "macro_auc": round(macro, 4)}


def plot_learning_curve() -> None:
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    loss = [row["train_loss"] for row in history]
    f1 = [row["valid_macro_f1"] for row in history]
    best = int(np.argmax(f1))

    fig, ax_loss = plt.subplots(figsize=(7.2, 5.2))
    ax_f1 = ax_loss.twinx()
    ax_loss.plot(epochs, loss, color="#1f77b4", marker="o", lw=1.8, label="train loss")
    ax_f1.plot(epochs, f1, color="#d62728", marker="o", lw=1.8, label="valid macro F1")
    ax_f1.scatter([epochs[best]], [f1[best]], color="#d62728", s=60, zorder=3)
    ax_f1.annotate(
        f"best  {f1[best]:.3f}",
        (epochs[best], f1[best]),
        textcoords="offset points",
        xytext=(8, -14),
        color="#d62728",
    )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    ax_f1.set_ylabel("valid macro F1")
    ax_f1.set_ylim(0.8, 1.0)
    ax_loss.set_title("분류 학습 곡선")
    lines = ax_loss.get_lines() + ax_f1.get_lines()
    ax_loss.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_9_2_2_learning_curve.png", dpi=160)
    plt.close(fig)


def plot_confusion() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    matrix = np.array(report["locked"]["confusion"], dtype=float)
    labels = [DISPLAY[name] for name in CLASSES]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title(f"분류 혼동행렬  (잠금 test, n={report['locked_n']})")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_9_2_3_confusion.png", dpi=160)
    plt.close(fig)


def plot_pr() -> dict:
    model = YOLO(str(DETECT_CKPT))
    metrics = model.val(
        data=str(DETECT_YAML),
        split="test",
        plots=False,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(OUT),
        name="det_test",
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    recall = np.asarray(box.px)
    precision = np.asarray(box.prec_values).reshape(-1)
    map50 = float(box.map50)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(recall, precision, color="#1f77b4", lw=2.2, label=f"wound  mAP@0.5 = {map50:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("재현율 (Recall)")
    ax.set_ylabel("정밀도 (Precision)")
    ax.set_title("탐지 Precision-Recall  (wound-detect test, IoU 0.5)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_9_2_4_pr.png", dpi=160)
    plt.close(fig)
    return {"map50": round(map50, 4), "map50_95": round(float(box.map), 4)}


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_learning_curve()
    plot_confusion()
    probs, targets = classification_probabilities()
    summary = {"roc": plot_roc(probs, targets), "detection": plot_pr()}
    (OUT / "figure_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
