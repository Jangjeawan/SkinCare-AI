"""서비스 토큰과 추론 동시 실행 수를 확인한다."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request


def verify_token(request: Request, authorization: str | None = Header(default=None)) -> None:
    token = request.app.state.settings["service_token"]
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail={"error": "unauthorized", "message": "서비스 토큰이 필요합니다."})


def acquire_inflight(request: Request) -> None:
    limit = request.app.state.settings["max_inflight"]
    if request.app.state.inflight >= limit:
        raise HTTPException(status_code=429, detail={"error": "busy", "message": "추론 대기 한도를 넘었습니다."})
    request.app.state.inflight += 1


def release_inflight(request: Request) -> None:
    request.app.state.inflight = max(0, request.app.state.inflight - 1)
