"""모델 적재 상태를 돌려준다. 토큰 값은 넣지 않는다."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from app.deps import verify_token
from guide.retrieve import load_cards
from qa.bank_loader import load_bank
from qa.llm_client import configured, model_name

VERSION = Path(__file__).resolve().parents[2] / "models" / "serving" / "version.json"

router = APIRouter(prefix="/ai", tags=["vision"], dependencies=[Depends(verify_token)])


@router.get("/health")
def health(request: Request):
    pipeline = request.app.state.pipeline
    ready = pipeline.ready and pipeline.energy_max is not None
    models = {"detect": "yolov8s-wound", "classify": "vit-b16-locked"}
    if VERSION.exists():
        saved = json.loads(VERSION.read_text(encoding="utf-8"))
        models = {"detect": saved.get("detect", models["detect"]), "classify": saved.get("classify", models["classify"])}
    return {
        "status": "ok" if ready else "degraded",
        "models": models,
        "bank_version": load_bank()["version"],
        "card_version": load_cards()["version"],
        "llm_model": model_name() if configured() else "unconfigured",
        "llm_configured": configured(),
        "inflight": request.app.state.inflight,
        "max_inflight": request.app.state.settings["max_inflight"],
    }
