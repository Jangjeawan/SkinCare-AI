from fastapi import APIRouter, Depends, HTTPException

from app.deps import verify_token
from app.schemas.qa import AnalyzeRequest
from qa.rules_engine import evaluate, load_rules

router = APIRouter(prefix="/ai", tags=["questionnaire"], dependencies=[Depends(verify_token)])


@router.post("/analyze")
def analyze(body: AnalyzeRequest):
    rules = load_rules()
    if not rules["rules"]:
        raise HTTPException(status_code=503, detail={"error": "rules_empty", "message": "응급 규칙이 아직 없습니다."})
    decision = evaluate(body.slots, rules, body.vision)
    return {
        "level": decision["level"],
        "rule_ids": decision["rule_ids"],
        "overrode_llm": False,
        "missing_slots": decision["missing_slots"],
        "scenario_source": "user" if body.slots.get("scenario") else None,
    }
