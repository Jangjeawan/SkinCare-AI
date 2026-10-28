"""문진 언어 모델. 키가 없으면 호출하지 않고, 실패하면 고정 질문으로 돌아간다."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

SYSTEM = (
    "출력은 JSON 한 개다. 키는 action, question_id, utterance, slots_delta, filled_from_user_text 만 쓴다. "
    "action은 ask 또는 finish다. question_id는 허용 목록 안의 문자열만 쓴다. "
    "utterance는 그 문항의 선택지를 쉬운 말로 묻는 한두 문장이다. "
    "근거가 없는 슬롯은 만들지 말고 키를 생략한다. "
    "진단 문장, 약물·연고 이름, 용량, 의료기관 이름은 넣지 않는다."
)


def model_name() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def build_messages(payload: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_content(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError("llm_invalid")
    return data


def complete(payload: dict, post=None) -> dict:
    if not configured() and post is None:
        raise RuntimeError("llm_unconfigured")
    body = {
        "model": model_name(),
        "temperature": 0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "messages": build_messages(payload),
    }
    sender = post or _post
    content = sender(body)
    return parse_content(content)


def _post(body: dict) -> str:
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("llm_unavailable") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("llm_invalid") from exc
