"""탐지 박스를 15% 늘려 이미지 경계 안에서 자른다."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: Path) -> np.ndarray | None:
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return None
    return np.asarray(image)


def read_bytes(raw: bytes) -> np.ndarray | None:
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        return None
    return np.asarray(image)


def expand_xywh(box: tuple[float, float, float, float], pad: float = 0.15) -> tuple[float, float, float, float]:
    x, y, w, h = box
    nx = x - w * pad
    ny = y - h * pad
    nw = w * (1 + 2 * pad)
    nh = h * (1 + 2 * pad)
    nx2 = min(1.0, nx + nw)
    ny2 = min(1.0, ny + nh)
    nx = max(0.0, nx)
    ny = max(0.0, ny)
    return nx, ny, nx2 - nx, ny2 - ny


def crop_box(image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.15, min_px: int = 32) -> np.ndarray | None:
    height, width = image.shape[:2]
    x, y, w, h = expand_xywh(box, pad)
    left = int(round(x * width))
    top = int(round(y * height))
    right = int(round((x + w) * width))
    bottom = int(round((y + h) * height))
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    if min(right - left, bottom - top) < min_px:
        return None
    return image[top:bottom, left:right]
