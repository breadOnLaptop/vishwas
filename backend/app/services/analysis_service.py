# analysis_service.py
from typing import Dict, Any, Optional, List

def compute_misinfo_score(
    text_signal: float,
    image_safe_search: Optional[float] = None,
    image_labels: Optional[List[str]] = None,
    ocr_text: str = "",
    llm_debug: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert signals into a 0..10 trustworthiness score (higher => more truthful).
    - text_signal: a 0..1 score from LLM/verification pipeline (1=most truthful)
    - image_safe_search: optional 0..1 image-safety score (1 = safe / trustworthy)
    - image_labels: optional labels extracted from image (unused in scoring but kept for audit)
    - ocr_text: optional OCR string detected in images (unused here)
    - llm_debug: optional dict of debug information (kept for logging / audits)

    Returns:
        {
            "score": float (0..10),
            "color": str (e.g. "red","amber","green"),
            "top_reasons": List[str]
        }
    """
    def clamp01(x: Optional[float]) -> float:
        try:
            if x is None:
                return 0.5
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.5

    text_val = clamp01(text_signal)
    image_safe_val = clamp01(image_safe_search)

    weight_text = 0.7
    weight_image = 0.3 if image_safe_search is not None else 0.0

    combined = text_val * weight_text + image_safe_val * weight_image
    score_0_10 = round(combined * 10.0, 2)

    # import here to avoid circular import at module load
    try:
        from app.services.google_ai_service import map_confidence_to_color
        color = map_confidence_to_color(combined)
    except Exception:
        # fallback mapping
        if combined <= 0.3:
            color = "red"
        elif combined <= 0.6:
            color = "orange"
        else:
            color = "green"

    reasons: List[str] = []
    if text_val <= 0.25:
        reasons.append(f"Text analysis strongly contradicts the claim (truthiness={text_val:.2f}).")
    elif text_val <= 0.5:
        reasons.append(f"Text analysis leans toward false/uncertain (truthiness={text_val:.2f}).")
    elif text_val < 0.85:
        reasons.append(f"Text analysis indicates some uncertainty (truthiness={text_val:.2f}).")
    else:
        reasons.append(f"Text analysis strongly supports the claim (truthiness={text_val:.2f}).")

    if image_safe_search is not None:
        if image_safe_val < 0.5:
            reasons.append(f"Image analysis indicates suspicious/unsafe content (safety={image_safe_val:.2f}).")
        else:
            reasons.append(f"Image analysis does not raise obvious safety issues (safety={image_safe_val:.2f}).")

    if llm_debug:
        reasons.append("LLM debug: see llm_debug for details.")

    if not reasons:
        reasons.append("No strong signals for misinformation detected; content appears credible.")

    return {"score": score_0_10, "color": color, "top_reasons ": reasons}
