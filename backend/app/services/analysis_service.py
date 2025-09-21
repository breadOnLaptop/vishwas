from typing import Dict, Any, Optional, List
from app.services.google_ai_service import map_confidence_to_color


def compute_misinfo_score(
    text_signal: float,
    image_safe_search: Optional[float] = None,
    image_labels: Optional[List[str]] = None,
    ocr_text: str = "",
    llm_debug: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert signals into a 0..10 trustworthiness score (higher => more truthful).

    Args:
        text_signal: 0..1 where 0 = misinformation, 1 = truthful
        image_safe_search: 0..1 where 0 = unsafe, 1 = safe. None -> treated as safe
        image_labels: list of image labels
        ocr_text: OCR-extracted text
        llm_debug: optional debug info

    Returns:
        dict with score (0..10), color (str), and top_reasons (list)
    """
    image_labels = image_labels or []

    image_safe_val = 1.0 if image_safe_search is None else max(0.0, min(1.0, float(image_safe_search)))
    text_val = max(0.0, min(1.0, float(text_signal)))

    # Weighted trust calculation
    w_text, w_image = 0.75, 0.25
    trustworthiness = max(0.0, min(1.0, w_text * text_val + w_image * image_safe_val))

    score_0_10 = round(trustworthiness * 10.0, 2)
    color = map_confidence_to_color(trustworthiness)

    reasons: List[str] = []
    # stronger messaging for definitive verdicts
    if text_val <= 0.3:
        reasons.append(f"Authoritative sources / fact-checks contradict this claim (truthiness={text_val:.2f}).")
    elif text_val < 0.85:
        reasons.append(f"Fact-check/LLM analysis indicates possible uncertainty (truthiness={text_val:.2f}).")
    if image_safe_val < 0.85:
        reasons.append(f"Image safety analysis indicates possible unsafe/suspicious content (safety={image_safe_val:.2f}).")
    if not reasons:
        reasons.append("No strong signals for misinformation detected; content appears credible.")

    return {"score": score_0_10, "color": color, "top_reasons": reasons}
