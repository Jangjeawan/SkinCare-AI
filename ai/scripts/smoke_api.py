"""헬스 한 번 호출. 토큰은 환경 변수 AI_SERVICE_TOKEN, 기본값은 dev-token."""

from __future__ import annotations

import os

import httpx

TOKEN = os.environ.get("AI_SERVICE_TOKEN", "dev-token")
URL = os.environ.get("AI_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    response = httpx.get(
        f"{URL}/api/v1/ai/health",
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=10,
    )
    response.raise_for_status()
    print(response.text)


if __name__ == "__main__":
    main()
