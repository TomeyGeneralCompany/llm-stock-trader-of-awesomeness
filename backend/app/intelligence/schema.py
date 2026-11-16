"""Typed research output contract for Ollama's native structured generation."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(StrictObject):
    chunk_id: str
    quote: str


class Finding(StrictObject):
    claim: str
    stance: Literal["supports", "contradicts", "unresolved"]
    evidence: list[Citation] = Field(min_length=1)


class Relationship(StrictObject):
    subject: str
    subject_kind: str
    object: str
    object_kind: str
    predicate: Literal["supplies", "purchases", "competes", "owns", "operates_in", "regulated_by", "depends_on", "exposed_to", "financed_by"]
    confidence: float = Field(ge=0, le=1)
    interpretation: str
    evidence: list[Citation] = Field(min_length=1)


class Contradiction(StrictObject):
    claim: str
    evidence: list[Citation] = Field(min_length=1)


class Hypothesis(StrictObject):
    event: str
    classification: str
    affected_symbols: list[str] = Field(min_length=1)
    direction: Literal[-1, 0, 1]
    expected_return: float = Field(ge=-1)
    horizon_minutes: int = Field(gt=0)
    causal_chain: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    assumptions: list[str] = Field(min_length=1)
    contradictions: list[Contradiction]
    invalidation: str
    evidence: list[Citation] = Field(min_length=1)


class Followup(StrictObject):
    question: str
    terms: str
    sources: list[str] = Field(min_length=1)


class ResearchOutput(StrictObject):
    status: Literal["sufficient", "continue", "exhausted"]
    findings: list[Finding]
    relationships: list[Relationship]
    hypotheses: list[Hypothesis]
    followups: list[Followup]
    error_classes: list[Literal["acquisition", "interpretation", "relationship", "directional", "magnitude", "timing", "allocation", "execution", "opportunity"]] = Field(default_factory=list)
