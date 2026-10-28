"""기존 잠금 지표를 검토한다. 요청 안에서 학습을 시작하지 않고, 가중치도 바꾸지 않는다."""

from __future__ import annotations

import json
from pathlib import Path

from jobs.queue_store import get, save

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "runs" / "classify_locked" / "test_metrics.json"
PASS_F1 = 0.80


def register(reason: str = "manual") -> dict:
    from jobs.queue_store import enqueue

    job = enqueue(reason)
    if job["status"] == "queued":
        return review(job["job_id"])
    return {"job_id": job["job_id"], "status": job["status"]}


def review(job_id: str, metrics_path: Path | None = None) -> dict:
    job = get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "missing"}
    if job["status"] != "queued":
        return {"job_id": job["job_id"], "status": job["status"]}
    path = metrics_path or DEFAULT_METRICS
    from jobs import queue_store

    log = queue_store.DIRECTORY / f"{job_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    job["status"] = "running"
    save(job)
    if not path.exists():
        log.write_text("잠금 지표 파일이 없어 학습 결과를 검토하지 못했다.\n", encoding="utf-8")
        job["status"] = "failed"
        job["log"] = str(log)
        save(job)
        return {"job_id": job_id, "status": "failed"}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    score = metrics.get("macro_f1")
    if score is None and isinstance(metrics.get("all"), dict):
        score = metrics["all"].get("macro_f1")
    job["status"] = "validating"
    job["macro_f1"] = score
    save(job)
    if not isinstance(score, (int, float)) or score < PASS_F1:
        log.write_text(f"macro_f1={score} 기준 {PASS_F1} 미만\n", encoding="utf-8")
        job["status"] = "failed"
        job["log"] = str(log)
        save(job)
        return {"job_id": job_id, "status": "failed"}
    log.write_text("기존 잠금 test 지표를 첨부했다. 새 학습은 돌리지 않았다.\n", encoding="utf-8")
    job["status"] = "awaiting_deploy"
    job["log"] = str(log)
    job["trained"] = False
    save(job)
    return {"job_id": job_id, "status": "awaiting_deploy"}
