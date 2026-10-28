"""YOLOv8s 단일 클래스 환부 탐지."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class WoundDetector:
    def __init__(self, weights: Path, conf: float, iou: float, max_det: int, device: str):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device

    def predict(self, image_rgb: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        prediction = self.model.predict(
            source=image_rgb[:, :, ::-1],
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
            device=self.device,
        )[0]
        if prediction.boxes is None or len(prediction.boxes) == 0:
            return []
        xywhn = prediction.boxes.xywhn.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        order = np.argsort(-scores)
        return [(tuple(map(float, xywhn[i])), float(scores[i])) for i in order]
