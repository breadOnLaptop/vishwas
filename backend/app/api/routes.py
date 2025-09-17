import io
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from pydantic import BaseModel
from app.schemas.analysis_schema import AnalysisResponse, RawVisionResult
from app.services.google_ai_service import (
    analyze_image_bytes,
    analyze_text_with_model,
    send_misinformation_report
)
from app.services.analysis_service import compute_misinfo_score

router = APIRouter()

# --- Image Analysis Endpoint ---
@router.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    vision_result = analyze_image_bytes(content)

    ocr_text = vision_result.get("text", "")

    prompt_text = f"Image filename: {file.filename}\n\nOCR extracted text:\n{ocr_text}\n\nDescribe up to 5 factual claims contained in the text and rate the likelihood they could be misinformation (0.0-1.0). Explain briefly why."

    llm_response = analyze_text_with_model(prompt_text)

    score_obj = compute_misinfo_score(
        text_signal=llm_response.get("misp_confidence", 0.0),
        image_safe_search=vision_result.get("safe_search_score", 0.0),
        image_labels=vision_result.get("labels", []),
        ocr_text=ocr_text
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": llm_response.get("explanation", ""),
        "vision": vision_result,
    }
    return JSONResponse(content=result)


# --- Text Analysis Endpoint ---
@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(text: str = Form(...), source_url: Optional[str] = Form(None)):
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty text")

    prompt_text = f"Extract up to 5 factual claims from the following text. For each claim return a veracity estimate (0.0-1.0) and 1-2 lines of reasoning. Text:\n\n{text}"

    llm_response = analyze_text_with_model(prompt_text)

    score_obj = compute_misinfo_score(
        text_signal=llm_response.get("misp_confidence", 0.0),
        image_safe_search=0.0,
        image_labels=[],
        ocr_text=text
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": llm_response.get("explanation", ""),
        "vision": None,
    }
    return JSONResponse(content=result)


# --- Report Endpoint ---
class ReportRequest(BaseModel):
    detected_content: str
    misinformation_score: float
    detected_source: str


@router.post("/report")
def report_misinformation(request: ReportRequest):
    threshold = 7.5
    if request.misinformation_score >= threshold:
        subject = "🚨 Misinformation Alert - Vishwas Prototype"
        body = (
            f"🚩 Potential Misinformation Detected\n\n"
            f"Content: {request.detected_content}\n"
            f"Score (0-10): {request.misinformation_score}\n"
            f"Source: {request.detected_source}\n\n"
            f"Review this content carefully."
        )
        authority_email = "your-real-email@example.com"  # Replace with your actual email
        send_misinformation_report(subject, body, authority_email)

        return {"status": "Report sent successfully", "score": request.misinformation_score}
    else:
        return {"status": "Content considered safe", "score": request.misinformation_score}
