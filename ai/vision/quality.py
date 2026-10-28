"""밝기·흐림·해상도 검사. 실패하면 탐지와 분류를 호출하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_side: int = 256
    luma_low: float = 40
    luma_high: float = 220
    blur_var: float = 80
    flat_std: float = 8


def check_image(image: np.ndarray, cfg: QualityConfig | None = None) -> dict:
    cfg = cfg or QualityConfig()
    if image is None or image.size == 0 or image.ndim != 3:
        return {"ok": False, "reasons": ["decode_fail"], "width": 0, "height": 0, "mean_luma": None, "lap_var": None}

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_luma = float(gray.mean())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if min(width, height) < cfg.min_side:
        reasons.append("too_small")
    if mean_luma < cfg.luma_low:
        reasons.append("too_dark")
    elif mean_luma > cfg.luma_high:
        reasons.append("too_bright")
    if float(gray.std()) < cfg.flat_std:
        reasons.append("flat")
    if lap_var < cfg.blur_var:
        reasons.append("blur")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "width": width,
        "height": height,
        "mean_luma": round(mean_luma, 2),
        "lap_var": round(lap_var, 2),
    }
