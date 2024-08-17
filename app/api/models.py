from pydantic import BaseModel
from typing import AsyncGenerator, Literal

from langchain.schema import Document

class InferenceRequest(BaseModel):
    session_id: str
    message: str

class SSEResponse(BaseModel):
    type: Literal["context", "start", "streaming", "end", "error"]
    value: str | list[Document]