from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.deps import acquire_inflight, release_inflight, verify_token
from tracking.compare import compare_images
from vision.preprocess import read_bytes

router = APIRouter(prefix="/ai", tags=["vision"], dependencies=[Depends(verify_token)])
MAX_BYTES = 8 * 1024 * 1024


@router.post("/tracking")
async def tracking(request: Request, before: UploadFile = File(...), after: UploadFile = File(...)):
    acquire_inflight(request)
    try:
        pipeline = request.app.state.pipeline
        if not pipeline.ready:
            raise HTTPException(status_code=503, detail={"error": "model_unavailable", "message": "탐지 모델을 불러오지 못했습니다."})
        images = []
        for upload in (before, after):
            raw = await upload.read()
            if len(raw) > MAX_BYTES:
                raise HTTPException(status_code=413, detail={"error": "image_too_large", "message": "이미지는 8MB 이하만 받는다."})
            image = read_bytes(raw)
            if image is None:
                raise HTTPException(status_code=415, detail={"error": "unsupported_media", "message": "이미지를 열 수 없습니다."})
            images.append(image)
        return compare_images(images[0], images[1], pipeline.detector.predict)
    finally:
        release_inflight(request)
