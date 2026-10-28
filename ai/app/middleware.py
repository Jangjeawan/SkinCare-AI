"""요청 ID와 한 줄 접속 로그. 본문과 이미지는 남기지 않는다."""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("skincare.access")


def access_record(request_id: str, method: str, path: str, status: int, latency_ms: int, text_chars: int | None) -> dict:
    return {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status,
        "latency_ms": latency_ms,
        "text_chars": text_chars,
    }


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        text_chars = None
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.body()
            text_chars = len(body.decode("utf-8", errors="ignore"))
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        record = access_record(
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            int((time.perf_counter() - started) * 1000),
            text_chars,
        )
        log.info(json.dumps(record, ensure_ascii=False))
        return response
