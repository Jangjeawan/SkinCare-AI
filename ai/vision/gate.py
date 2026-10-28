"""신뢰도 게이트. 조건을 하나라도 놓치면 다섯 유형 대신 UNKNOWN을 낸다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ["abrasion", "bruise", "burn", "open_wound", "normal_skin"]
DISPLAY = {
    "abrasion": "찰과상 의심",
    "bruise": "타박상 의심",
    "burn": "화상 의심",
    "open_wound": "벌어진 상처 의심",
    "normal_skin": "뚜렷한 손상 특징이 보이지 않음",
    "UNKNOWN": "사진만으로 구분하기 어렵습니다. 다시 촬영하거나 의료기관에 문의하세요",
}


@dataclass(frozen=True)
class GateConfig:
    top1: float = 0.65
    margin: float = 0.15
    det_conf: float = 0.35
    min_area: float = 0.02


def energy_score(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    peak = logits.max()
    return float(-(peak + np.log(np.exp(logits - peak).sum())))


def apply_gate(
    probs: np.ndarray | None,
    det_score: float | None,
    area_ratio: float | None,
    energy: float | None,
    energy_max: float | None,
    cfg: GateConfig | None = None,
) -> dict:
    cfg = cfg or GateConfig()
    fails: list[str] = []
    if det_score is None or area_ratio is None or area_ratio < cfg.min_area:
        fails.append("no_roi")
    elif det_score < cfg.det_conf:
        fails.append("low_det")

    margin = None
    probabilities = None
    if probs is None:
        fails.append("low_conf")
    else:
        probs = np.asarray(probs, dtype=np.float64)
        order = np.argsort(-probs)
        top1 = float(probs[order[0]])
        margin = float(probs[order[0]] - probs[order[1]])
        probabilities = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
        if top1 < cfg.top1:
            fails.append("low_conf")
        if margin < cfg.margin:
            fails.append("low_margin")

    ood = energy_max is None or energy is None or energy > energy_max
    if ood:
        fails.append("ood")

    label = "UNKNOWN" if fails else CLASSES[int(np.argsort(-probs)[0])]
    return {
        "label": label,
        "display": DISPLAY[label],
        "probabilities": probabilities,
        "margin": None if margin is None else round(margin, 4),
        "ood": ood,
        "gate_fail": fails,
    }
