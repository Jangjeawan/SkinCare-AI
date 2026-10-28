"""카드 색인. 출처 있는 카드만 남긴다."""

from __future__ import annotations

from guide.retrieve import load_cards


def build_index() -> dict:
    return load_cards()
