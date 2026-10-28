"""환경 변수와 config.yaml을 합친다."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def settings() -> dict:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["service_token"] = os.environ.get("AI_SERVICE_TOKEN", "dev-token")
    config["max_inflight"] = int(os.environ.get("MAX_INFLIGHT", "2"))
    return config
