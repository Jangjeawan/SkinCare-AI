"""재학습 잡 상태. 디스크에 남기고 deployed로는 승인 스크립트만 넘긴다."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "var" / "jobs"
OPEN = {"queued", "running", "validating", "awaiting_deploy"}


def _path(job_id: str) -> Path:
    return DIRECTORY / f"{job_id}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(job: dict) -> dict:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    _path(job["job_id"]).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def enqueue(reason: str) -> dict:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path in DIRECTORY.glob("*.json"):
        job = _read(path)
        if job.get("status") in OPEN:
            return job
    return _write({"job_id": str(uuid.uuid4()), "status": "queued", "reason": reason, "trained": False})


def get(job_id: str) -> dict | None:
    path = _path(job_id)
    if not path.exists():
        return None
    return _read(path)


def save(job: dict) -> dict:
    return _write(job)
