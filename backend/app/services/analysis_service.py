from typing import List, Dict, Any

def compute_misinfo_score(text_signal: float, image_safe_search: float, image_labels: List[str], ocr_text: str) -> Dict[str, Any]:
    """
    Combine signals into a 0-10 score and color mapping.
    Inputs:
      - text_signal: 0..1 (LLM-estimated misinfo likelihood)
      - image_safe_search: 0..1 (derived from Vision safesearch)
      - image_labels: list of labels (strings)
      - ocr_text: extracted text
    Returns:
      { score: float (0-10), color: str, top_reasons: [strings] }
    """
    # weights (same as earlier design)
    w_text = 0.45
    w_image = 0.20
    w_label = 0.10
    w_ocr_mismatch = 0.25

    # label heuristic: if label set contains 'satire', 'parody' reduce risk; if 'deepfake' or 'manipulated' increase
    label_score = 0.0
    label_lower = [l.lower() for l in (image_labels or [])]
    if any("deepfake" in l or "manipulated" in l or "photoshopped" in l or "edited" in l for l in label_lower):
        label_score = 1.0
    elif any("satire" in l or "parody" in l for l in label_lower):
        label_score = 0.0
    else:
        label_score = 0.2 if len(label_lower) > 0 else 0.0

    # ocr mismatch heuristic: if OCR text looks like sensational patterns, increase
    ocr_score = 0.0
    if ocr_text:
        low = ocr_text.lower()
        if "shocking" in low or "you won't believe" in low or "exclusive" in low or "miracle" in low:
            ocr_score = 1.0
        elif len(low) < 20:
            ocr_score = 0.05

    combined = (
        w_text * float(text_signal)
        + w_image * float(image_safe_search)
        + w_label * float(label_score)
        + w_ocr_mismatch * float(ocr_score)
    )

    # clamp and scale 0..10
    score_0_10 = max(0.0, min(10.0, combined * 10.0))

    # color mapping
    if score_0_10 <= 3.3:
        color = "green"
    elif score_0_10 <= 6.6:
        color = "yellow"
    else:
        color = "red"

    # top reasons (human-readable)
    reasons = []
    reasons.append(f"LLM estimate: {text_signal:.2f}")
    reasons.append(f"Image safe-search score: {image_safe_search:.2f}")
    if label_score > 0.5:
        reasons.append("Image labels indicate possible manipulation or deepfake.")
    if ocr_score > 0.5:
        reasons.append("OCR contains sensational phrasing.")

    return {"score": round(score_0_10, 2), "color": color, "top_reasons": reasons}
