from fastapi import APIRouter, Depends

from app.deps import verify_token
from app.schemas.qa import FeedbackRequest
from jobs.feedback_store import enqueue

router = APIRouter(prefix="/ai", tags=["mlops"], dependencies=[Depends(verify_token)])


@router.post("/feedback")
def feedback(body: FeedbackRequest):
    queue_id = enqueue(body.model_dump())
    return {"queued": True, "queue_id": queue_id}
