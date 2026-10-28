"""병명 단정과 상품명이 섞인 발화는 슬롯으로 받지 않는다."""

from __future__ import annotations

import re

_DRUG = re.compile(r"(연고|정제|캡슐|mg|밀리그램)", re.IGNORECASE)


def reject_utterance(text: str) -> str | None:
    if _DRUG.search(text or ""):
        return "drug_mention"
    return None
