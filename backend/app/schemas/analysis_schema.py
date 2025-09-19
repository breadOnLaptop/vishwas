# app/schemas/analysis_schema.py
from typing import Any, List, Optional, Dict
from pydantic import BaseModel

class VisionResult(BaseModel):
    text: str
    labels: List[str]
    safe_search: Dict[str, Any]
    safe_search_score: Optional[float]
    raw: Dict[str, Any]

class ClaimReference(BaseModel):
    title: Optional[str]
    link: Optional[str]
    snippet: Optional[str]

class Claim(BaseModel):
    text: str
    misp_confidence: float
    short_reason: Optional[str]
    references: Optional[List[ClaimReference]] = []

class PerClaimSources(BaseModel):
    claim: str
    sources: List[ClaimReference] = []

class AnalysisResponse(BaseModel):
    score: float
    color: str
    top_reasons: List[str]
    explanation: str
    vision: Optional[VisionResult] = None
    parsed: Optional[Dict[str, Any]] = None  # leave as free-form, contains 'claims' list
    sources: Optional[List[Dict[str, Any]]] = []  # per-claim sources
    top_sources: Optional[List[Dict[str, Any]]] = []  # flat list of title/link/snippet
    debug: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "score": 8.5,
                "color": "green",
                "top_reasons": ["LLM estimate indicates safe", "No suspicious image signals"],
                "explanation": "{...}",
                "vision": {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "raw": {}},
                "parsed": {"claims": []},
                "sources": [],
                "top_sources": [],
                "debug": {}
            }
        }
