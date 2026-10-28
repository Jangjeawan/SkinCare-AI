"""AI 서빙 진입점. 외부에는 8000을 열지 않는다."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import settings
from app.middleware import RequestIdMiddleware
from app.routers import analyze, feedback, guide, health, predict, question, retrain, tracking
from vision.pipeline import DEFAULT_ENERGY, VisionPipeline

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings()
    app.state.inflight = 0
    app.state.tmp = ROOT / "tmp"
    app.state.tmp.mkdir(exist_ok=True)
    pipeline = VisionPipeline(app.state.settings)
    if DEFAULT_ENERGY.exists():
        pipeline.energy_max = json.loads(DEFAULT_ENERGY.read_text(encoding="utf-8"))["energy_max"]
    try:
        pipeline.load()
    except Exception:
        pipeline.detector = None
        pipeline.classifier = None
    app.state.pipeline = pipeline
    yield


app = FastAPI(title="SkinCare AI", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.include_router(predict.router, prefix="/api/v1")
app.include_router(question.router, prefix="/api/v1")
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(guide.router, prefix="/api/v1")
app.include_router(tracking.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(retrain.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
