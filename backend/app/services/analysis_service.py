from typing import Dict, Any, Optional
from .google_ai_service import map_confidence_to_color

def compute_misinfo_score(
    text_signal: float,
    image_safe_search: Optional[float] = None,
    image_labels: list | None = None,
    ocr_text: str = "",
    llm_debug: dict | None = None,
) -> Dict[str, Any]:
    """
    Convert signals into a 0..10 misinfo score (higher => more likely misinformation).
    text_signal: 0..1 where 0 = misinformation, 1 = truthful/safe
    image_safe_search: 0..1 where 0 = unsafe, 1 = safe.
        If None -> no image was provided -> treat as fully safe (1.0).
    """
    if image_labels is None:
        image_labels = []

    # If no image provided treat as fully safe (doesn't contribute to misinfo)
    if image_safe_search is None:
        image_safe_search_val = 1.0
    else:
        # clamp to [0,1]
        image_safe_search_val = max(0.0, min(1.0, float(image_safe_search)))

    # misinfo_from_text: 0..1 where 1 means definitely misinformation
    misinfo_from_text = 1.0 - max(0.0, min(1.0, float(text_signal)))

    # misinfo_from_image: 0..1 (if image unsafe => increases misinfo)
    misinfo_from_image = 1.0 - image_safe_search_val

    # weights (tuneable)
    w_text = 0.75
    w_image = 0.25

    misinfo_likelihood = (w_text * misinfo_from_text) + (w_image * misinfo_from_image)
    misinfo_likelihood = max(0.0, min(1.0, misinfo_likelihood))

    score_0_10 = round(misinfo_likelihood * 10.0, 2)

    # color mapping uses original text_signal (0..1 where low = misinformation)
    color = map_confidence_to_color(float(text_signal))

    # top reasons assembly (improved ordering)
    reasons = []
    if misinfo_from_text > 0.05:
        reasons.append(f"LLM estimate indicates possible misinformation (likelihood={misinfo_from_text:.2f})")
    if misinfo_from_image > 0.05:
        reasons.append(f"Image safe-search signals suspicious content (likelihood={misinfo_from_image:.2f})")
    if not reasons:
        reasons.append("No strong signals for misinformation detected.")

    return {"score": score_0_10, "color": color, "top_reasons": reasons}
