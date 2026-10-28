from fastapi import APIRouter, Depends

from app.deps import verify_token
from app.schemas.qa import GuideRequest
from guide.retrieve import load_cards, select

router = APIRouter(prefix="/ai", tags=["guide"], dependencies=[Depends(verify_token)])


@router.post("/guide")
def guide(body: GuideRequest):
    cards = load_cards()
    picked = select(body.level, body.slots, body.rule_ids, cards)
    picked["card_version"] = cards["version"]
    return picked
