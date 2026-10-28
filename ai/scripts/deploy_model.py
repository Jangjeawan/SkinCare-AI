"""승인된 잡의 가중치만 서빙 폴더로 옮긴다. 호출하지 않으면 교체하지 않는다."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.queue_store import get, save

def approve(job_id: str, classify_src: Path, detect_src: Path, serving_dir: Path, archive_dir: Path) -> dict:
    job = get(job_id)
    if job is None or job.get("status") != "awaiting_deploy":
        raise RuntimeError("awaiting_deploy 상태가 아니면 가중치를 교체하지 않는다.")
    if not classify_src.exists() or not detect_src.exists():
        raise RuntimeError("교체할 가중치 파일이 없다.")
    serving_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for name in ("classify.pt", "detect.pt"):
        current = serving_dir / name
        if current.exists():
            shutil.copy2(current, archive_dir / f"{stamp}-{name}")
    shutil.copy2(classify_src, serving_dir / "classify.pt")
    shutil.copy2(detect_src, serving_dir / "detect.pt")
    version = {"detect": f"yolov8s-wound-{job_id[:8]}", "classify": f"vit-b16-locked-{job_id[:8]}", "job_id": job_id}
    (serving_dir / "version.json").write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    job["status"] = "deployed"
    save(job)
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve:
        raise SystemExit("승인 플래그 없이는 교체하지 않는다.")
    version = approve(
        args.job_id,
        ROOT / "runs" / "classify_locked" / "best.pt",
        ROOT / "runs" / "det_v8s_wound" / "weights" / "best.pt",
        ROOT / "models" / "serving",
        ROOT / "models" / "archive",
    )
    print(json.dumps(version, ensure_ascii=False))


if __name__ == "__main__":
    main()
