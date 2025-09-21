from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field, root_validator


class VisionResult(BaseModel):
    text: str = ""
    labels: List[str] = Field(default_factory=lambda: [])
    safe_search: Dict[str, Any] = Field(default_factory=lambda: {})
    safe_search_score: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=lambda: {})


class ClaimReference(BaseModel):
    title: Optional[str] = ""
    link: Optional[str] = ""
    snippet: Optional[str] = ""
    publisher: Optional[str] = ""


class Claim(BaseModel):
    text: str
    misp_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    misp_confidence_0_1: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_0_10: Optional[float] = Field(None, ge=0.0, le=10.0)
    short_reason: Optional[str] = ""
    references: List[ClaimReference] = Field(default_factory=lambda: [])

    @root_validator(pre=True)
    def _normalize_confidences(cls, values):
        m01 = values.get("misp_confidence_0_1")
        legacy = values.get("misp_confidence")
        c10 = values.get("confidence_0_10")

        chosen: Optional[float] = None
        if m01 is not None:
            chosen = float(m01)
        elif legacy is not None:
            chosen = float(legacy)
        elif c10 is not None:
            chosen = float(c10) / 10.0

        if chosen is None:
            chosen = 0.5
        chosen = max(0.0, min(1.0, chosen))

        values["misp_confidence_0_1"] = chosen
        values["misp_confidence"] = chosen
        values["confidence_0_10"] = round(chosen * 10.0, 2)

        if values.get("references") is None:
            values["references"] = []
        return values


class ParsedOut(BaseModel):
    overall_misp_confidence_0_1: Optional[float] = Field(None, ge=0.0, le=1.0)
    claims: List[Claim] = Field(default_factory=lambda: [])


class AnalysisResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=10.0)
    color: str
    top_reasons: List[str] = Field(default_factory=lambda: [])
    user_explanation: str = ""
    top_sources: List[ClaimReference] = Field(default_factory=lambda: [])

    explanation: Optional[str] = ""
    vision: Optional[VisionResult] = None
    parsed: Optional[ParsedOut] = None
    sources: Optional[List[Dict[str, Any]]] = Field(default_factory=lambda: [])
    top_sources_legacy: Optional[List[Dict[str, Any]]] = Field(default_factory=lambda: [])
    debug: Optional[Dict[str, Any]] = None
