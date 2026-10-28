"""문항 뱅크를 메모리에 올린다. 문항이 없으면 버전만 돌려준다."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_bank(path: Path | None = None) -> dict:
    path = path or ROOT / "qa" / "bank.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"version": data.get("version", "unset"), "questions": data.get("questions") or []}
