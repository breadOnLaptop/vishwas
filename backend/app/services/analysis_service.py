# analysis_service.py
from typing import Dict, Any, Optional, List
from .google_ai_service import map_confidence_to_color

def compute_misinfo_score(
    text_signal: float,
    image_safe_search: Optional[float] = None,
    image_labels: Optional[List[str]] = None,
    ocr_text: str = "",
    llm_debug: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Convert signals into a 0..10 *trustworthiness* score (higher => more likely truthful/good).
    - text_signal: 0..1 where 0 = misinformation, 1 = truthful/safe
    - image_safe_search: 0..1 where 0 = unsafe, 1 = safe.
        If None -> no image was provided -> treat as fully safe (1.0).
    """
    if image_labels is None:
        image_labels = []

    # If no image provided treat as fully safe (doesn't contribute negatively)
    image_safe_search_val = 1.0 if image_safe_search is None else max(0.0, min(1.0, float(image_safe_search)))

    # text_signal is 0..1 where 1 = truthful. Use directly as truth signal.
    text_signal_val = max(0.0, min(1.0, float(text_signal)))

    # weights (tuneable) - give text more weight
    w_text = 0.75
    w_image = 0.25

    # trustworthiness (0..1) -> higher = better/true
    trustworthiness = (w_text * text_signal_val) + (w_image * image_safe_search_val)
    trustworthiness = max(0.0, min(1.0, trustworthiness))

    # convert to 0..10 (higher is good/truthful)
    score_0_10 = round(trustworthiness * 10.0, 2)

    # color mapping uses trustworthiness (1=truthy -> green)
    color = map_confidence_to_color(float(trustworthiness))

    # top reasons assembly (phrasing changed to reflect trustiness)
    reasons: List[str] = []
    if text_signal_val < 0.85:
        reasons.append(f"LLM estimate indicates potential uncertainty (truthiness={text_signal_val:.2f})")
    if image_safe_search_val < 0.85:
        reasons.append(f"Image safe-search signals possible unsafe/suspicious content (safety={image_safe_search_val:.2f})")
    if not reasons:
        reasons.append("No strong signals for misinformation detected; content appears credible based on current signals.")

    return {"score": score_0_10, "color": color, "top_reasons": reasons}
