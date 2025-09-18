from pydantic import BaseModel
from typing import Any, List, Optional, Dict

class RawVisionResult(BaseModel):
    text: Optional[str] = None
    labels: Optional[List[str]] = []
    safe_search: Optional[Dict[str, Any]] = {}
    raw: Optional[Dict[str, Any]] = {}

class SourceItem(BaseModel):
    title: Optional[str]
    link: Optional[str]
    snippet: Optional[str]

class ClaimSources(BaseModel):
    claim: str
    sources: List[SourceItem] = []

class AnalysisResponse(BaseModel):
    score: float
    color: str
    top_reasons: List[str]
    explanation: str
    vision: Optional[RawVisionResult]
    sources: Optional[List[Dict[str, Any]]] = []
