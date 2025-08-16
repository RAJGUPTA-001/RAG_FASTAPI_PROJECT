from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

ModelName = Literal["openai/gpt-oss-120b"]
DEFAULT_MODEL: ModelName = "openai/gpt-oss-120b"


class QueryInput(BaseModel):
    question: str
    session_id: str | None = Field(default=None)
    model: ModelName = Field(default=DEFAULT_MODEL)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName = Field(default=DEFAULT_MODEL)


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int
