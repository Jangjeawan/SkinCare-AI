from fastapi import APIRouter, Depends, HTTPException, Request

from app.deps import verify_token
from app.schemas.qa import QuestionRequest
from qa.bank_loader import load_bank
from qa.filter import reject_utterance
from qa.llm_client import complete, configured
from qa.orchestrator import handle_turn
from qa.rules_engine import load_rules

router = APIRouter(prefix="/ai", tags=["questionnaire"], dependencies=[Depends(verify_token)])


def _llm(payload: dict) -> dict:
    return complete(payload)


@router.post("/question")
def question(body: QuestionRequest, request: Request):
    if reject_utterance(body.user_text):
        raise HTTPException(status_code=422, detail={"error": "drug_mention", "message": "약품 이름이 포함된 발화는 받지 않습니다."})
    bank = load_bank()
    if body.bank_version and body.bank_version != bank["version"]:
        raise HTTPException(status_code=422, detail={"error": "invalid_body", "message": "문항 뱅크 버전이 다릅니다."})
    if not bank["questions"]:
        raise HTTPException(status_code=503, detail={"error": "bank_empty", "message": "문항 뱅크가 아직 없습니다."})
    rules = load_rules()
    if not rules["rules"]:
        raise HTTPException(status_code=503, detail={"error": "rules_empty", "message": "응급 규칙이 아직 없습니다."})
    caller = _llm if configured() else None
    result = handle_turn(body.model_dump(), bank, rules, caller)
    result["request_id"] = getattr(request.state, "request_id", None)
    return result
