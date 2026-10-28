"""ViT-B/16 다섯 유형 분류. 현재 가중치는 사진 전체로 학습되어 있다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_classify import load_model
from train_classify import build_transforms
from vision.gate import energy_score


class WoundClassifier:
    def __init__(self, weights: Path, device: torch.device):
        self.device = device
        self.model = load_model(weights, device)
        self.transform = build_transforms(False)

    def logits(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor = self.transform(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor)[0].float().cpu().numpy()

    def probabilities(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        logits = self.logits(image_rgb)
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        probs = exp / exp.sum()
        return probs, energy_score(logits)
