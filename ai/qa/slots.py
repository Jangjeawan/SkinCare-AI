"""슬롯은 허용 값만 남긴다. 시간 표현은 표가 언어 모델보다 우선한다."""

from __future__ import annotations

SLOT_VALUES = {
    "scenario": {"burn", "open_wound", "abrasion", "bruise", "unsure"},
    "elapsed": {"lt_1h", "h1_to_72", "gt_72h", "unknown"},
    "cause": {"heat_liquid", "flame", "chemical", "electric", "unknown"},
    "site": {"face", "hand", "joint", "genitalia", "wide", "other", "unknown"},
    "bleeding": {"stopped", "ongoing", "unknown"},
    "wound_open": {"yes", "no", "unknown"},
    "foreign_body": {"yes", "no", "unknown"},
    "dirty": {"soil", "rust", "animal", "clean", "unknown"},
    "blister": {"yes", "no", "unknown"},
    "stuck_clothes": {"yes", "no", "unknown"},
    "systemic": {"breath", "swelling_fast", "fever", "none", "unknown"},
    "red_spread": {"yes", "no", "unknown"},
    "pus": {"yes", "no", "unknown"},
    "pain_trend": {"up", "same", "down", "unknown"},
    "tetanus_memory": {"remember_yes", "remember_no", "unknown"},
}

_ELAPSED = (
    ("lt_1h", ("방금", "조금 전", "10분", "한 시간 안", "1시간 안")),
    ("h1_to_72", ("어제", "이틀", "하루")),
    ("gt_72h", ("사흘", "일주일", "지난주", "3일")),
    ("unknown", ("모르겠어요", "언제인지 모르", "아까인지")),
)
_CAUSE = (
    ("chemical", ("화학",)),
    ("electric", ("전기", "감전")),
    ("heat_liquid", ("뜨거운 물", "뜨거운 기름", "끓는")),
    ("flame", ("불꽃", "화염", "불에")),
)
_BLEEDING = (
    ("ongoing", ("피가 안 멈", "피가 계속", "지혈이 안")),
    ("stopped", ("피는 멈", "피가 멈")),
)
_SITE = (
    ("wide", ("손바닥보다", "넓은 범위", "넓게")),
    ("face", ("얼굴", "안면")),
    ("joint", ("관절",)),
    ("genitalia", ("성기", "회음", "생식기")),
    ("hand", ("손",)),
)


def empty(value) -> bool:
    return value is None or value == "" or value == []


def normalize_systemic(value):
    if value is None:
        return None
    items = [value] if isinstance(value, str) else list(value)
    items = [item for item in items if item in SLOT_VALUES["systemic"]]
    if any(item != "none" for item in items):
        items = [item for item in items if item != "none"]
    return items or None


def accept(key: str, value):
    if key == "systemic":
        return normalize_systemic(value)
    allowed = SLOT_VALUES.get(key)
    if allowed is None or value not in allowed:
        return None
    return value


def _one(text: str, table: tuple) -> str | None:
    found = {value for value, phrases in table if any(phrase in text for phrase in phrases)}
    if len(found) != 1:
        return None
    return next(iter(found))


def read_user_text(text: str) -> dict:
    """발화에서 근거가 하나인 슬롯만 채운다. 모순이면 그 슬롯은 비운다."""
    text = text or ""
    found = {}
    elapsed = _one(text, _ELAPSED)
    if elapsed:
        found["elapsed"] = elapsed
    cause = _one(text, _CAUSE)
    if cause:
        found["cause"] = cause
    bleeding = _one(text, _BLEEDING)
    if bleeding:
        found["bleeding"] = bleeding
    site = _one(text, _SITE)
    if site:
        found["site"] = site
    systemic = []
    if any(phrase in text for phrase in ("숨쉬기 어렵", "숨이 차", "호흡이")):
        systemic.append("breath")
    if any(phrase in text for phrase in ("갑자기 붓", "심하게 붓")):
        systemic.append("swelling_fast")
    if "열이" in text:
        systemic.append("fever")
    if systemic:
        found["systemic"] = systemic
    return found


def merge(slots: dict, delta: dict) -> tuple[dict, dict]:
    merged = dict(slots or {})
    applied = {}
    for key, value in (delta or {}).items():
        cleaned = accept(key, value)
        if cleaned is None or not empty(merged.get(key)):
            continue
        merged[key] = cleaned
        applied[key] = cleaned
    return merged, applied
