"""피드백은 메타만 큐 파일로 남긴다. 다시 추론하지 않는다."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "var" / "feedback"


def enqueue(payload: dict, directory: Path | None = None) -> str:
    folder = directory or DIRECTORY
    folder.mkdir(parents=True, exist_ok=True)
    queue_id = str(uuid.uuid4())
    record = {
        "queue_id": queue_id,
        "request_id": payload.get("request_id"),
        "kind": payload.get("kind"),
        "note": payload.get("note"),
        "photo_ref": payload.get("photo_ref"),
    }
    (folder / f"{queue_id}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return queue_id
