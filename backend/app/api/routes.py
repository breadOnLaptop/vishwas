# app/api/routes.py
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

# For document parsing
from io import BytesIO
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

router = APIRouter()


@router.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    vision_result = analyze_image_bytes(content)
    ocr_text = vision_result.get("text", "") or ""

    unified = analyze_content(text=ocr_text, image_bytes=content, source_url=source_url)

    misp_confidence = float(unified.get("misp_confidence", 0.5))
    explanation = unified.get("explanation", "")
    parsed = unified.get("parsed", None)
    sources = unified.get("sources", [])            # per-claim sources
    top_sources = unified.get("top_sources", [])    # flat clickable links
    debug = unified.get("debug", {})

    # Use None for image_safe_search when vision didn't provide a score
    image_safe_search_score = vision_result.get("safe_search_score") if vision_result.get("safe_search_score") is not None else None

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=image_safe_search_score,
        image_labels=vision_result.get("labels", []),
        ocr_text=ocr_text,
        llm_debug=debug
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": explanation,
        "vision": vision_result,
        "parsed": parsed,
        "sources": sources,
        "top_sources": top_sources,
        "debug": debug,   # remove in production if not desired
    }
    return JSONResponse(content=result)


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(text: str = Form(...), source_url: Optional[str] = Form(None)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    unified = analyze_content(text=text, source_url=source_url)

    misp_confidence = float(unified.get("misp_confidence", 0.5))
    explanation = unified.get("explanation", "")
    parsed = unified.get("parsed", None)
    sources = unified.get("sources", [])
    top_sources = unified.get("top_sources", [])
    debug = unified.get("debug", {})

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=None,
        image_labels=[],
        ocr_text=text,
        llm_debug=debug
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": explanation,
        "vision": None,
        "parsed": parsed,
        "sources": sources,
        "top_sources": top_sources,
        "debug": debug,
    }
    return JSONResponse(content=result)


class ReportRequest(BaseModel):
    detected_content: str
    misinformation_score: float
    detected_source: str


@router.post("/report")
def report_misinformation(request: ReportRequest):
    # report when the score is LOW (score: 0..10 where HIGHER = more truthful)
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
        authority_email = "peeyushmaurya.dev@gmail.com"  # Replace
        send_misinformation_report(subject, body, authority_email)
        return {"status": "Report sent successfully", "score": request.misinformation_score}
    else:
        return {"status": "Content considered safe", "score": request.misinformation_score}


@router.post("/analyze/document", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...), source_url: Optional[str] = Form(None)):
    """
    Accepts PDFs, DOCX and plain text files. Extracts text and runs the same analyze_content flow.
    Note: this endpoint currently extracts text only. If you need OCR of embedded images in PDFs,
    use pdf2image + analyze_image_bytes, or use Google Vision Document/Batch API for PDF OCR.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = (file.filename or "").lower()
    extracted_text = ""

    # PDF
    if filename.endswith(".pdf") and PdfReader is not None:
        try:
            reader = PdfReader(BytesIO(content))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    # ignore page extraction errors, continue
                    pages.append("")
            extracted_text = "\n".join(pages).strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {e}")

    # DOCX
    elif filename.endswith(".docx") and docx is not None:
        try:
            doc = docx.Document(BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text]
            extracted_text = "\n".join(paragraphs).strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DOCX text extraction failed: {e}")

    else:
        # Try to decode as UTF-8 / fallback to latin-1 plain text
        try:
            extracted_text = content.decode("utf-8").strip()
        except Exception:
            try:
                extracted_text = content.decode("latin-1").strip()
            except Exception:
                # Unknown binary file: attempt to run OCR via Vision (document bytes as image)
                # This is best-effort — if it's an image file (e.g., scanned PDF converted to single image),
                # analyze_image_bytes will run Vision OCR.
                vision_result = analyze_image_bytes(content)
                extracted_text = vision_result.get("text", "") or ""

    if not extracted_text:
        # If no text extracted, still run analyze_content (fallback will try to create claims from context)
        extracted_text = ""

    unified = analyze_content(text=extracted_text, image_bytes=None, source_url=source_url)

    misp_confidence = float(unified.get("misp_confidence", 0.5))
    explanation = unified.get("explanation", "")
    parsed = unified.get("parsed", None)
    sources = unified.get("sources", [])
    top_sources = unified.get("top_sources", [])
    debug = unified.get("debug", {})

    score_obj = compute_misinfo_score(
        text_signal=misp_confidence,
        image_safe_search=None,
        image_labels=[],
        ocr_text=extracted_text,
        llm_debug=debug
    )

    result = {
        "score": score_obj["score"],
        "color": score_obj["color"],
        "top_reasons": score_obj["top_reasons"],
        "explanation": explanation,
        "vision": None,
        "parsed": parsed,
        "sources": sources,
        "top_sources": top_sources,
        "debug": debug,
    }
    return JSONResponse(content=result)
