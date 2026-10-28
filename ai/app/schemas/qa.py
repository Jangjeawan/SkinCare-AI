from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    session_id: str | None = None
    bank_version: str | None = None
    user_text: str = ""
    slots: dict = Field(default_factory=dict)
    vision: dict = Field(default_factory=dict)
    turn: int = 0
    last_question_id: str | None = None


class AnalyzeRequest(BaseModel):
    session_id: str | None = None
    slots: dict = Field(default_factory=dict)
    vision: dict = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    request_id: str
    kind: str
    note: str | None = None
    photo_ref: str | None = None


class GuideRequest(BaseModel):
    level: str | None = None
    rule_ids: list[str] = Field(default_factory=list)
    slots: dict = Field(default_factory=dict)
    bank_version: str | None = None
