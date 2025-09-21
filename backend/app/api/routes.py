import json
import logging
from io import BytesIO
from typing import Optional, Any, Dict

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.schemas.analysis_schema import AnalysisResponse
from app.services.google_ai_service import (
    analyze_image_bytes,
    analyze_content,
    report_if_severe,
    send_misinformation_report,
)
from app.services.analysis_service import compute_misinfo_score
from app.core.config import settings

# Optional imports for document parsing
PdfReader: Optional[Any] = None
docx: Optional[Any] = None

try:
    from PyPDF2 import PdfReader as _PdfReader  # type: ignore
    PdfReader = _PdfReader
except Exception:
    PdfReader = None

try:
    import docx as _docx  # python-docx
    docx = _docx
except Exception:
    docx = None

logger = logging.getLogger("routes")
router = APIRouter()


def _ensure_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return list(v.values()) if v else []
    return [v]


def _derive_misp_confidence_from_unified(unified: Dict[str, Any]) -> float:
    try:
        parsed = unified.get("parsed") or {}
        if isinstance(parsed, dict):
            for k in ["overall_misp_confidence", "overall_misp_confidence_0_1"]:
                if parsed.get(k) is not None:
                    val = float(parsed[k])
                    return max(0.0, min(1.0, val))
            claims = parsed.get("claims") or []
            vals = []
            for cl in claims:
                if isinstance(cl, dict):
                    v = cl.get("misp_confidence") or cl.get("misp_confidence_0_1")
                    if v is not None:
                        vals.append(float(v))
            if vals:
                return sum(vals) / len(vals)
        score = unified.get("score")
        if score is not None:
            return max(0.0, min(1.0, float(score) / 10.0))
    except Exception:
        logger.debug("Failed to derive misp_confidence", exc_info=True)
    return 0.5


@router.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    vision_result = analyze_image_bytes(content)
    ocr_text = vision_result.get("text", "") or ""
    unified = analyze_content(text=ocr_text, image_bytes=content, source_url=source_url)
    misp_confidence = _derive_misp_confidence_from_unified(unified)

    sources = _ensure_list(unified.get("sources", []))
    top_sources = _ensure_list(unified.get("top_sources", []))

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=vision_result.get("safe_search_score"),
        image_labels=vision_result.get("labels", []),
        ocr_text=ocr_text,
        llm_debug=unified.get("debug"),
    )

    result = {
        "score": score_obj.get("score"),
        "color": score_obj.get("color"),
        "top_reasons": score_obj.get("top_reasons"),
        "explanation": unified.get("explanation", "") or json.dumps(unified.get("parsed", {}))[:5000],
        "user_explanation": unified.get("user_explanation", ""),
        "vision": vision_result,
        "parsed": unified.get("parsed", {"claims": []}),
        "sources": sources,
        "top_sources": top_sources,
        "debug": unified.get("debug", {}),
    }

    try:
        u = dict(unified)
        u["misp_confidence"] = misp_confidence
        report_if_severe(u, report_to_email=getattr(settings, "REPORT_TO_EMAIL", None))
    except Exception as e:
        logger.warning(f"report_if_severe failed: {e}", exc_info=True)

    return JSONResponse(content=result)


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(text: str = Form(...), source_url: Optional[str] = Form(None)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    unified = analyze_content(text=text, source_url=source_url)
    misp_confidence = _derive_misp_confidence_from_unified(unified)

    sources = _ensure_list(unified.get("sources", []))
    top_sources = _ensure_list(unified.get("top_sources", []))

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=None,
        image_labels=[],
        ocr_text=text,
        llm_debug=unified.get("debug"),
    )

    result = {
        "score": score_obj.get("score", unified.get("score", 5.0)),
        "color": score_obj.get("color", unified.get("color", "orange")),
        "top_reasons": score_obj.get("top_reasons", unified.get("top_reasons", [])),
        "explanation": unified.get("explanation", "") or json.dumps(unified.get("parsed", {}))[:5000],
        "user_explanation": unified.get("user_explanation", ""),
        "vision": None,
        "parsed": unified.get("parsed", {"claims": []}),
        "sources": sources,
        "top_sources": top_sources,
        "debug": unified.get("debug", {}),
    }

    try:
        u = dict(unified)
        u["misp_confidence"] = misp_confidence
        report_if_severe(u, report_to_email=getattr(settings, "REPORT_TO_EMAIL", None))
    except Exception as e:
        logger.warning(f"report_if_severe failed: {e}", exc_info=True)

    return JSONResponse(content=result)


class ReportRequest(BaseModel):
    detected_content: str
    misinformation_score: float
    detected_source: str


@router.post("/report")
def report_misinformation(request: ReportRequest):
    threshold = 2.5
    if request.misinformation_score <= threshold:
        subject = "🚨 Misinformation Alert - Vishwas Prototype"
        body = (
            f"🚩 Potential Misinformation Detected\n\n"
            f"Content: {request.detected_content}\n"
            f"Score (0-10, higher=more truthful): {request.misinformation_score}\n"
            f"Source: {request.detected_source}\n\n"
            f"Review this content carefully."
        )
        authority_email = getattr(settings, "REPORT_TO_EMAIL", None) or "peeyushmaurya.dev@gmail.com"
        send_misinformation_report(subject, body, authority_email)
        return {"status": "Report sent successfully", "score": request.misinformation_score}
    return {"status": "Content considered safe", "score": request.misinformation_score}


@router.post("/analyze/document", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = (file.filename or "").lower()
    extracted_text = ""

    if filename.endswith(".pdf") and PdfReader is not None:
        reader = PdfReader(BytesIO(content))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        extracted_text = "\n".join(pages).strip()

    elif filename.endswith(".docx") and docx is not None:
        from docx import Document  # type: ignore
        doc = Document(BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        extracted_text = "\n".join(paragraphs).strip()

    else:
        try:
            extracted_text = content.decode("utf-8").strip()
        except Exception:
            try:
                extracted_text = content.decode("latin-1").strip()
            except Exception:
                vision_result = analyze_image_bytes(content)
                extracted_text = vision_result.get("text", "") or ""

    unified = analyze_content(text=extracted_text, image_bytes=None, source_url=source_url)
    misp_confidence = _derive_misp_confidence_from_unified(unified)

    sources = _ensure_list(unified.get("sources", []))
    top_sources = _ensure_list(unified.get("top_sources", []))

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=None,
        image_labels=[],
        ocr_text=extracted_text,
        llm_debug=unified.get("debug"),
    )

    result = {
        "score": score_obj.get("score", unified.get("score", 5.0)),
        "color": score_obj.get("color", unified.get("color", "orange")),
        "top_reasons": score_obj.get("top_reasons", unified.get("top_reasons", [])),
        "explanation": unified.get("explanation", "") or json.dumps(unified.get("parsed", {}))[:5000],
        "user_explanation": unified.get("user_explanation", ""),
        "vision": None,
        "parsed": unified.get("parsed", {"claims": []}),
        "sources": sources,
        "top_sources": top_sources,
        "debug": unified.get("debug", {}),
    }

    try:
        u = dict(unified)
        u["misp_confidence"] = misp_confidence
        report_if_severe(u, report_to_email=getattr(settings, "REPORT_TO_EMAIL", None))
    except Exception as e:
        logger.warning(f"report_if_severe failed: {e}", exc_info=True)

    return JSONResponse(content=result)
