from pydantic import BaseModel


class ErrorBody(BaseModel):
    error: str
    message: str
    request_id: str | None = None
