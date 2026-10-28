"""두 사진의 환부 면적 비율과 크롭 평균 색을 비교한다. 임상 문장은 만들지 않는다."""

from __future__ import annotations

import numpy as np

from vision.preprocess import crop_box

NOTE = "촬영 거리·조명이 같다는 보정이 없다. 참고값이다."


def trend(before_area: float | None, after_area: float | None) -> tuple[str, float | None]:
    if before_area is None or after_area is None or before_area <= 0 or after_area <= 0:
        return "uncomparable", None
    delta = after_area - before_area
    if abs(delta) < 0.02:
        return "similar", delta
    return ("larger" if delta > 0 else "smaller"), delta


def mean_hsv(crop: np.ndarray) -> dict[str, float]:
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float64)
    return {
        "hue": float(hsv[:, :, 0].mean()),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "value": float(hsv[:, :, 2].mean() / 255.0),
    }


def measure(image: np.ndarray, boxes: list) -> dict | None:
    if not boxes:
        return None
    box, score = boxes[0]
    _x, _y, width, height = box
    crop = crop_box(image, tuple(box), pad=0.0, min_px=8)
    if crop is None:
        return None
    return {"area_ratio": float(width * height), "score": float(score), "hsv": mean_hsv(crop)}


def compare_images(before: np.ndarray, after: np.ndarray, detect) -> dict:
    before_view = measure(before, detect(before))
    after_view = measure(after, detect(after))
    if before_view is None or after_view is None:
        return {"area_ratio_delta": None, "color_delta": None, "trend_hint": "uncomparable", "note": NOTE}
    hint, delta = trend(before_view["area_ratio"], after_view["area_ratio"])
    color = {key: round(after_view["hsv"][key] - before_view["hsv"][key], 4) for key in ("hue", "saturation", "value")}
    return {
        "area_ratio_delta": None if delta is None else round(delta, 4),
        "color_delta": color,
        "trend_hint": hint,
        "note": NOTE,
    }
