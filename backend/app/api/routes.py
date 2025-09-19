import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from pydantic import BaseModel
from app.schemas.analysis_schema import AnalysisResponse
from app.services.google_ai_service import (
    analyze_image_bytes,
    analyze_content,
    send_misinformation_report,
)
from app.services.analysis_service import compute_misinfo_score

router = APIRouter()

@router.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    vision_result = analyze_image_bytes(content)
    ocr_text = vision_result.get("text", "") or ""

    unified = analyze_content(text=ocr_text, image_bytes=content)

    misp_confidence = float(unified.get("misp_confidence", 0.5))
    explanation = unified.get("explanation", "")
    sources = unified.get("sources", [])

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=vision_result.get("safe_search_score", 0.0),
        image_labels=vision_result.get("labels", []),
        ocr_text=ocr_text,
        llm_debug=unified.get("debug")
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": explanation,
        "vision": vision_result,
        "sources": sources
    }
    return JSONResponse(content=result)


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(text: str = Form(...), source_url: Optional[str] = Form(None)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    unified = analyze_content(text=text)

    misp_confidence = float(unified.get("misp_confidence", 0.5))
    explanation = unified.get("explanation", "")
    sources = unified.get("sources", [])

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=None,
        image_labels=[],
        ocr_text=text,
        llm_debug=unified.get("debug")
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": explanation,
        "vision": None,
        "sources": sources
    }
    return JSONResponse(content=result)

class ReportRequest(BaseModel):
    detected_content: str
    misinformation_score: float
    detected_source: str

@router.post("/report")
def report_misinformation(request: ReportRequest):
    # Previously threshold was 7.5 (when higher=more likely misinformation).
    # After flipping the 0..10 scale so HIGHER = more truthful, we report when score is LOW.
    threshold = 2.5  # equivalent to previous 7.5 in the old scale

    if request.misinformation_score <= threshold:
        subject = "🚨 Misinformation Alert - Vishwas Prototype"
        body = (
            f"🚩 Potential Misinformation Detected\n\n"
            f"Content: {request.detected_content}\n"
            f"Score (0-10, higher=more truthful): {request.misinformation_score}\n"
            f"Source: {request.detected_source}\n\n"
            f"Review this content carefully."
        )
        authority_email = "peeyushmaurya.dev@gmail.com"  # Replace with real authority
        send_misinformation_report(subject, body, authority_email)
        return {"status": "Report sent successfully", "score": request.misinformation_score}
    else:
        return {"status": "Content considered safe", "score": request.misinformation_score}
