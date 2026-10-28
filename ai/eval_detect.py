"""환부 탐지 가중치를 valid에서 후처리 설정을 고른 뒤 test에 한 번 적용한다.

설정 후보: NMS IoU, 박스 축소 비율, TTA, 여러 가중치의 WBF 앙상블.
클래스는 무시하고 환부 박스 하나로 본다(single_cls 평가).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def load_split(root: Path, split: str):
    images = sorted((root / "images" / split).glob("*.jpg"))
    gts = []
    label_dir = root / "labels" / split
    for img in images:
        rows = []
        label_path = label_dir / f"{img.stem}.txt"
        text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        for line in text.splitlines():
            if line.strip():
                _, x, y, w, h = map(float, line.split()[:5])
                rows.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        gts.append(np.array(rows) if rows else np.zeros((0, 4)))
    return images, gts


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def shrink_boxes(boxes: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0 or len(boxes) == 0:
        return boxes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0]) * factor
    h = (boxes[:, 3] - boxes[:, 1]) * factor
    return np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), 0, 1)


def average_precision(preds, gts, iou_thr: float = 0.5) -> float:
    scores, matched = [], []
    n_gt = sum(len(g) for g in gts)
    for (boxes, conf), gt in zip(preds, gts):
        if len(boxes) == 0:
            continue
        m = iou_matrix(boxes, gt) if len(gt) else np.zeros((len(boxes), 0))
        used = set()
        for i in np.argsort(-conf):
            hit = False
            if m.shape[1]:
                j = int(m[i].argmax())
                if m[i, j] >= iou_thr and j not in used:
                    used.add(j)
                    hit = True
            scores.append(conf[i])
            matched.append(hit)
    if not scores:
        return 0.0
    order = np.argsort(-np.array(scores))
    matched = np.array(matched)[order]
    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def wbf(list_of_preds, iou_thr: float = 0.55, skip: float = 0.0):
    """단순 Weighted Box Fusion. list_of_preds[model][image] = (boxes, conf)."""
    n_models = len(list_of_preds)
    fused_all = []
    for image_index in range(len(list_of_preds[0])):
        boxes, confs = [], []
        for model_preds in list_of_preds:
            b, c = model_preds[image_index]
            keep = c >= skip
            boxes.append(b[keep])
            confs.append(c[keep])
        boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4))
        confs = np.concatenate(confs) if confs else np.zeros(0)
        order = np.argsort(-confs)
        clusters: list[list[int]] = []
        cluster_boxes: list[np.ndarray] = []
        for i in order:
            placed = False
            for k, cb in enumerate(cluster_boxes):
                if iou_matrix(boxes[i : i + 1], cb[None])[0, 0] >= iou_thr:
                    clusters[k].append(i)
                    members = clusters[k]
                    weights = confs[members]
                    cluster_boxes[k] = (boxes[members] * weights[:, None]).sum(0) / weights.sum()
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                cluster_boxes.append(boxes[i].copy())
        out_boxes = np.array(cluster_boxes) if cluster_boxes else np.zeros((0, 4))
        out_conf = np.array(
            [confs[members].sum() / n_models for members in clusters]
        ) if clusters else np.zeros(0)
        fused_all.append((out_boxes, out_conf))
    return fused_all


def predict(model: YOLO, images, nms_iou: float, tta: bool):
    results = model.predict(
        source=[str(p) for p in images],
        conf=0.001,
        iou=nms_iou,
        augment=tta,
        verbose=False,
        device=0,
        max_det=100,
    )
    preds = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            preds.append((np.zeros((0, 4)), np.zeros(0)))
        else:
            preds.append((r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(r"c:\Users\kirit\졸프\datasets\wound-detect"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    valid_images, valid_gts = load_split(args.data, "valid")
    test_images, test_gts = load_split(args.data, "test")
    models = [YOLO(str(w)) for w in args.weights]

    nms_options = [0.5, 0.6, 0.7]
    shrink_options = [1.0, 0.95, 0.9, 0.85]
    tta_options = [False, True]

    report = {"weights": [str(w) for w in args.weights], "valid": {}, "test": {}}
    cache = {}
    best = (-1.0, None)
    for model_index, model in enumerate(models):
        for nms_iou, tta in itertools.product(nms_options, tta_options):
            preds = predict(model, valid_images, nms_iou, tta)
            cache[(model_index, nms_iou, tta)] = preds
            for factor in shrink_options:
                adjusted = [(shrink_boxes(b, factor), c) for b, c in preds]
                ap = average_precision(adjusted, valid_gts)
                key = f"m{model_index}_nms{nms_iou}_tta{int(tta)}_shrink{factor}"
                report["valid"][key] = round(ap, 4)
                if ap > best[0]:
                    best = (ap, ("single", model_index, nms_iou, tta, factor))
    if len(models) > 1:
        for nms_iou, tta in itertools.product(nms_options, tta_options):
            fused = wbf([cache[(i, nms_iou, tta)] for i in range(len(models))])
            for factor in shrink_options:
                adjusted = [(shrink_boxes(b, factor), c) for b, c in fused]
                ap = average_precision(adjusted, valid_gts)
                key = f"wbf_nms{nms_iou}_tta{int(tta)}_shrink{factor}"
                report["valid"][key] = round(ap, 4)
                if ap > best[0]:
                    best = (ap, ("wbf", None, nms_iou, tta, factor))

    report["best_valid"] = {"ap50": round(best[0], 4), "config": best[1]}
    kind, model_index, nms_iou, tta, factor = best[1]

    # test: baseline (first model, default) and chosen config, each once
    base = predict(models[0], test_images, 0.7, False)
    report["test"]["baseline_m0_default"] = round(average_precision(base, test_gts), 4)
    if kind == "single":
        preds = predict(models[model_index], test_images, nms_iou, tta)
    else:
        preds = wbf([predict(m, test_images, nms_iou, tta) for m in models])
    adjusted = [(shrink_boxes(b, factor), c) for b, c in preds]
    report["test"]["chosen"] = round(average_precision(adjusted, test_gts), 4)
    report["test"]["chosen_ap30"] = round(average_precision(adjusted, test_gts, 0.3), 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_valid": report["best_valid"], "test": report["test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
