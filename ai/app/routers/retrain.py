from fastapi import APIRouter, Depends

from app.deps import verify_token
from jobs.queue_store import get
from jobs.retrain_worker import register

router = APIRouter(prefix="/ai", tags=["mlops"], dependencies=[Depends(verify_token)])


@router.post("/retrain")
def retrain(reason: str = "manual"):
    return register(reason)


@router.get("/jobs/{job_id}")
def job(job_id: str):
    found = get(job_id)
    if found is None:
        return {"job_id": job_id, "status": "missing"}
    return found
