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
    - text_signal: 0..1 where 0 = misinformation, 1 = truthful
    - image_safe_search: 0..1 where 0 = unsafe, 1 = safe. If None -> treat as safe (1.0).
    Returns:
      {"score": float (0..10), "color": str, "top_reasons": List[str]}
    """
    if image_labels is None:
        image_labels = []

    image_safe_search_val = 1.0 if image_safe_search is None else max(0.0, min(1.0, float(image_safe_search)))
    text_signal_val = max(0.0, min(1.0, float(text_signal)))

    # weights (tunable)
    w_text = 0.75
    w_image = 0.25

    trustworthiness = (w_text * text_signal_val) + (w_image * image_safe_search_val)
    trustworthiness = max(0.0, min(1.0, trustworthiness))

    # convert to 0..10 (higher is good/truthful)
    score_0_10 = round(trustworthiness * 10.0, 2)

    # color mapping uses trustworthiness (1=truthy -> green)
    color = map_confidence_to_color(float(trustworthiness))

    # top reasons assembly
    reasons: List[str] = []
    if text_signal_val < 0.85:
        reasons.append(f"LLM/fact-check estimate indicates potential uncertainty (truthiness={text_signal_val:.2f})")
    if image_safe_search_val < 0.85:
        reasons.append(f"Image safe-search indicates potential unsafe/suspicious content (safety={image_safe_search_val:.2f})")
    if not reasons:
        reasons.append("No strong signals for misinformation detected; content appears credible based on fact-checks and other signals.")

    return {"score": score_0_10, "color": color, "top_reasons": reasons}
