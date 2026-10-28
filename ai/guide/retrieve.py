"""카드는 조건이 맞은 원문만 반환한다. 주소가 없거나 본문이 비면 뺀다."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cards(path: Path | None = None) -> dict:
    path = path or ROOT / "guide" / "cards.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cards = [card for card in (data.get("cards") or []) if card.get("source_url") and card.get("body")]
    return {"version": data.get("version", "unset"), "cards": cards}


def load_avoid_labels(path: Path | None = None) -> dict:
    path = path or ROOT / "guide" / "avoid_labels.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_denylist(path: Path | None = None) -> list[str]:
    path = path or ROOT / "guide" / "denylist.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _when(when: dict, level: str, slots: dict, rule_ids: list[str]) -> bool:
    if "any" in when:
        return any(_when(item, level, slots, rule_ids) for item in when["any"])
    if "scenario" in when and slots.get("scenario") != when["scenario"]:
        return False
    if "scenario_in" in when and slots.get("scenario") not in when["scenario_in"]:
        return False
    if "cause" in when and slots.get("cause") != when["cause"]:
        return False
    if "cause_in" in when and slots.get("cause") not in when["cause_in"]:
        return False
    if "level" in when and level != when["level"]:
        return False
    if "level_in" in when and level not in when["level_in"]:
        return False
    if "level_not" in when and level == when["level_not"]:
        return False
    if "rule" in when and when["rule"] not in rule_ids:
        return False
    return True


def _redact(body: str, banned: list[str]) -> tuple[str, int]:
    parts = re.split(r"(?<=[\.!?。])\s+|\n+", body)
    kept = []
    removed = 0
    for part in parts:
        if not part.strip():
            continue
        if any(word in part for word in banned):
            removed += 1
            continue
        kept.append(part.strip())
    return " ".join(kept), removed


def select(level: str | None, slots: dict, rule_ids: list[str], cards: dict, labels: dict | None = None, banned: list[str] | None = None) -> dict:
    if not level:
        return {"cards": [], "avoid": [], "reason": "no_level", "redactions": 0}
    labels = labels if labels is not None else load_avoid_labels()
    banned = banned if banned is not None else load_denylist()
    matched = []
    for card in cards.get("cards", []):
        if _when(card.get("when") or {}, level, slots or {}, rule_ids or []):
            matched.append(card)
    dropped = {item for card in matched for item in (card.get("replaces") or [])}
    matched = [card for card in matched if card.get("id") not in dropped][:3]
    redactions = 0
    visible = []
    avoid_ids = []
    for card in matched:
        body, removed = _redact(card.get("body") or "", banned)
        redactions += removed
        if not body:
            continue
        visible.append(
            {
                "id": card.get("id"),
                "title": card.get("title"),
                "body": body,
                "source": card.get("source"),
                "source_url": card.get("source_url"),
            }
        )
        avoid_ids.extend(card.get("avoid") or [])
    avoid = [{"id": key, "label": labels[key]} for key in labels if key in set(avoid_ids)]
    return {
        "cards": visible,
        "avoid": avoid,
        "reason": None if visible else "no_card",
        "redactions": redactions,
    }
