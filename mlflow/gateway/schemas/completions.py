from typing import List, Optional

from pydantic import BaseModel, Extra

from .chat import CommonRequestPayload


class RequestPayload(CommonRequestPayload):
    prompt: str


class CandidateMetadata(BaseModel, extra=Extra.forbid):
    finish_reason: Optional[str]


class Candidate(BaseModel, extra=Extra.forbid):
    text: str
    metadata: CandidateMetadata


class Metadata(BaseModel, extra=Extra.forbid):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: str


class ResponsePayload(BaseModel, extra=Extra.allow):
    candidates: List[Candidate]
    metadata: Metadata
