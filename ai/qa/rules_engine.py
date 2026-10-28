"""규칙 카탈로그. 응급은 먼저 맞은 하나에서 멈추고, 경고는 모은다."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_REQUIRED = {
    "burn": ["elapsed", "cause", "site"],
    "open_wound": ["elapsed", "bleeding"],
    "abrasion": ["elapsed", "site"],
    "bruise": ["elapsed", "site", "systemic"],
    "unsure": ["elapsed", "systemic"],
}


def load_rules(path: Path | None = None) -> dict:
    path = path or ROOT / "qa" / "rules.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = sorted(data.get("rules") or [], key=lambda item: item.get("priority", 99))
    return {"version": data.get("version", "unset"), "rules": rules}


def missing_slots(slots: dict) -> list[str]:
    scenario = slots.get("scenario")
    if not scenario:
        return ["scenario"]
    if scenario == "burn" and slots.get("cause") in {"chemical", "electric"}:
        return []
    if scenario == "open_wound" and slots.get("bleeding") == "ongoing":
        return []
    needed = list(_REQUIRED.get(scenario, []))
    if scenario == "abrasion" and slots.get("elapsed") == "gt_72h":
        needed.extend(["red_spread", "pus", "pain_trend"])
    return [key for key in needed if slots.get(key) in (None, "", [])]


def _systemic(slots: dict) -> list:
    value = slots.get("systemic")
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def matches(when: dict, slots: dict, vision: dict | None) -> bool:
    vision = vision or {}
    if "any" in when and not any(matches(item, slots, vision) for item in when["any"]):
        return False
    for key, expected in when.items():
        if key == "any":
            continue
        if key in {"scenario", "cause", "site", "bleeding", "wound_open", "blister", "elapsed", "stuck_clothes", "foreign_body", "red_spread", "pus", "pain_trend"}:
            if slots.get(key) != expected:
                return False
        elif key == "systemic_has":
            if expected not in _systemic(slots):
                return False
        elif key == "dirty_in":
            if slots.get("dirty") not in expected:
                return False
        elif key == "vision_label":
            if vision.get("label") != expected or not vision.get("gate_passed"):
                return False
        elif key == "gate_passed":
            if bool(vision.get("gate_passed")) != bool(expected):
                return False
        elif key == "depth_class":
            hint = vision.get("depth_hint") or {}
            score = float(hint.get("score") or 0)
            if str(hint.get("raw_class")) != str(expected):
                return False
            if score < float(when.get("depth_score_gte", 0)):
                return False
        elif key == "depth_score_gte":
            continue
        else:
            return False
    return True


def evaluate(slots: dict, rules: dict, vision: dict | None = None) -> dict:
    slots = slots or {}
    vision = vision or {}
    emergency = [rule for rule in rules.get("rules", []) if rule.get("level") == "EMERGENCY"]
    warning = [rule for rule in rules.get("rules", []) if rule.get("level") == "WARNING"]
    for rule in emergency:
        if matches(rule.get("when") or {}, slots, vision):
            return {"level": "EMERGENCY", "rule_ids": [rule["id"]], "missing_slots": []}
    hits = [rule["id"] for rule in warning if matches(rule.get("when") or {}, slots, vision)]
    absent = missing_slots(slots)
    if hits:
        return {"level": "WARNING", "rule_ids": hits, "missing_slots": absent}
    if absent:
        return {"level": None, "rule_ids": [], "missing_slots": absent}
    if vision.get("gate_passed") is False:
        return {"level": "UNKNOWN", "rule_ids": ["vision.unknown"], "missing_slots": []}
    return {"level": "OBSERVATION", "rule_ids": ["default.observe"], "missing_slots": []}
