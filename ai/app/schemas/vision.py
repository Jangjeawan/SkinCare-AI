from pydantic import BaseModel


class DetectionBox(BaseModel):
    box: list[float]
    score: float


class ClassificationBody(BaseModel):
    label: str
    display: str
    probabilities: dict[str, float] | None = None
    margin: float | None = None
    ood: bool
    gate_fail: list[str]


class PredictResponse(BaseModel):
    request_id: str
    quality: dict
    detection: DetectionBox | None = None
    extra_boxes: list[list[float]]
    classification: ClassificationBody | None = None
    depth_hint: None = None
    area_ratio: float | None = None
    model_versions: dict[str, str]
