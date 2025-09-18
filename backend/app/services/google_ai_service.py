import os
import re
import json
import logging
from typing import Dict, Any, Optional, List

from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings

logger = logging.getLogger("google_ai_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
    logger.info(f"Using GOOGLE_APPLICATION_CREDENTIALS: {settings.GOOGLE_APPLICATION_CREDENTIALS}")
else:
    logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set or file does not exist; Vision/Vertex may be disabled.")

# Vision client init
VISION_CLIENT = None
try:
    from google.cloud import vision
    if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
        creds = service_account.Credentials.from_service_account_file(settings.GOOGLE_APPLICATION_CREDENTIALS)
        VISION_CLIENT = vision.ImageAnnotatorClient(credentials=creds)
        logger.info("Vision client initialized.")
    else:
        VISION_CLIENT = None
        logger.warning("Vision client disabled due to missing credentials.")
except Exception as e:
    VISION_CLIENT = None
    vision = None
    logger.warning(f"google-cloud-vision import/init failed: {e}")

# Vertex init (try both generative_models and language_models)
vertex_available = False
_vertex_model_api = None
try:
    try:
        from vertexai import init as vertex_init
        from vertexai.generative_models import GenerativeModel
        vertex_creds = None
        if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
            vertex_creds = service_account.Credentials.from_service_account_file(settings.GOOGLE_APPLICATION_CREDENTIALS)
        vertex_init(project=settings.GCP_PROJECT, location=settings.GCP_REGION, credentials=vertex_creds)
        vertex_available = True
        _vertex_model_api = "generative_models"
        logger.info("Vertex (generative_models) initialized.")
    except Exception as e1:
        try:
            from vertexai import init as vertex_init2
            from vertexai.language_models import TextGenerationModel
            vertex_creds = None
            if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
                vertex_creds = service_account.Credentials.from_service_account_file(settings.GOOGLE_APPLICATION_CREDENTIALS)
            vertex_init2(project=settings.GCP_PROJECT, location=settings.GCP_REGION, credentials=vertex_creds)
            vertex_available = True
            _vertex_model_api = "language_models"
            logger.info("Vertex (language_models) initialized.")
        except Exception as e2:
            vertex_available = False
            logger.warning(f"Vertex init failed (both paths): {e1} / {e2}")
except Exception as e:
    vertex_available = False
    logger.warning(f"Vertex import/init overall failed: {e}")

# Custom Search helper (with clearer error handling)
def _search_web_google_customsearch(query: str, num: int = 3) -> List[Dict[str, str]]:
    api_key = settings.GOOGLE_SEARCH_API_KEY
    cx = settings.GOOGLE_SEARCH_CX
    if not api_key or not cx:
        logger.debug("Custom Search API not configured; returning empty sources.")
        return []

    try:
        from googleapiclient.discovery import build
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cx, num=min(num, 10)).execute()
        items = res.get("items", [])[:num]
        out = []
        for it in items:
            out.append({"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")})
        return out
    except Exception as e:
        logger.debug(f"googleapiclient failed: {e}. Trying HTTP fallback.")
        try:
            import httpx
            params = {"key": api_key, "cx": cx, "q": query, "num": str(num)}
            r = httpx.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                items = data.get("items", [])[:num]
                return [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]
            else:
                # clearer message for invalid key or other bad args
                logger.warning(f"Custom Search HTTP call failed ({r.status_code}): {r.text}")
        except Exception as e2:
            logger.warning(f"Custom Search HTTP fallback failed: {e2}")
    return []

# helpers
def _parse_number_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([01](?:\.\d{1,3})?)", text)
    if m:
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            return None
    return None

def _vertex_parse_structured(text_out: str) -> Optional[Dict[str, Any]]:
    if not text_out:
        return None
    start = text_out.find("{")
    end = text_out.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text_out[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                candidate_fixed = candidate.replace("'", "\"")
                return json.loads(candidate_fixed)
            except Exception:
                return None
    return None

# --- improved normalization for parsed confidences ---
_truth_keywords = [
    "true", "accurate", "accurately", "fundamental", "universally accepted",
    "scientific consensus", "supported by", "evidence", "proven", "confirmed",
    "established", "accepted", "indeed", "correct", "correctly", "accurate",
    "global pandemic", "pandemic", "pandemics", "widespread", "widespreadly",
    "caused", "causes", "cause", "resulted in", "resulted", "led to", "led",
    "significant", "numerous", "health", "economic", "social", "foundational"
]

_false_keywords = [
    "no evidence", "debunk", "debunked", "false", "misleading", "fabricated",
    "conspiracy", "untrue", "incorrect", "not true", "disproved", "hoax",
    "fake", "unsupported", "refuted"
]

def _normalize_parsed_confidences(parsed: Dict[str, Any], explanation_text: str) -> (Dict[str, Any], Optional[float]):
    """
    Inspect parsed JSON and the LLM explanation text; if textual cues contradict numeric confidences,
    fix them. Returns (updated_parsed, corrected_overall_confidence).
    Preference rules:
      - If textual signal is strong (truth vs false) for a claim and numeric deviates significantly (>0.4),
        prefer textual signal.
      - If numeric not provided, infer from textual cues.
      - Overall is average of corrected claim confidences if available.
    """
    if not parsed or not isinstance(parsed, dict):
        return parsed, None

    claims = parsed.get("claims", [])
    corrected_scores: List[float] = []

    # Lowercased context: explanation + top_reasons (if any)
    context_parts = []
    if explanation_text:
        context_parts.append(explanation_text.lower())
    top_reasons = parsed.get("top_reasons", [])
    if isinstance(top_reasons, list):
        context_parts.extend([str(r).lower() for r in top_reasons])
    context_text = " ".join(context_parts)

    for idx, c in enumerate(claims):
        # guard
        if not isinstance(c, dict):
            c = {"text": str(c)}

        claim_text = (c.get("text") or "").lower()
        short_reason = (c.get("short_reason") or "").lower()
        combined = " ".join([context_text, claim_text, short_reason]).lower()

        # count truth vs false signals
        t_count = sum(1 for kw in _truth_keywords if kw in combined)
        f_count = sum(1 for kw in _false_keywords if kw in combined)

        # textual_inferred_conf: prefer 1.0 if truth signals > false signals, 0.0 if vice versa, else None
        textual_inferred_conf: Optional[float] = None
        if t_count > f_count and t_count >= 1:
            textual_inferred_conf = 1.0
        elif f_count > t_count and f_count >= 1:
            textual_inferred_conf = 0.0

        numeric = c.get("misp_confidence", None)
        corrected: float

        if isinstance(numeric, (int, float)):
            numeric = float(numeric)
            # if we have a textual inference and it contradicts numeric strongly -> use textual
            if textual_inferred_conf is not None and abs(numeric - textual_inferred_conf) > 0.4:
                corrected = float(textual_inferred_conf)
                logger.debug(
                    f"Normalized claim[{idx}] numeric {numeric} -> {corrected} due to textual cues (t_count={t_count}, f_count={f_count})."
                )
            else:
                # numeric seems consistent or no strong textual signals -> keep numeric
                corrected = numeric
        else:
            # no numeric provided; use textual inference if any, otherwise default 0.5
            if textual_inferred_conf is not None:
                corrected = float(textual_inferred_conf)
            else:
                corrected = 0.5

        # write back corrected value
        try:
            c["misp_confidence"] = float(corrected)
        except Exception:
            c["misp_confidence"] = corrected

        corrected_scores.append(float(c["misp_confidence"]))

    # compute overall as average of corrected claim confidences if available
    overall = None
    if corrected_scores:
        overall = sum(corrected_scores) / len(corrected_scores)
        parsed["overall_misp_confidence"] = overall

    return parsed, overall

# Vision analyze
def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    if VISION_CLIENT is None:
        return {"text": "", "labels": [], "safe_search": {}, "safe_search_score": 0.0, "raw": {}}
    try:
        from google.cloud import vision
        image = vision.Image(content=image_bytes)
        response = VISION_CLIENT.annotate_image({
            "image": image,
            "features": [
                {"type_": vision.enums.Feature.Type.TEXT_DETECTION},
                {"type_": vision.enums.Feature.Type.LABEL_DETECTION},
                {"type_": vision.enums.Feature.Type.SAFE_SEARCH_DETECTION},
            ],
        })
        text = ""
        if getattr(response, "text_annotations", None):
            try:
                text = response.text_annotations[0].description or ""
            except Exception:
                text = ""
        labels = [ann.description for ann in (getattr(response, "label_annotations", []) or [])]
        safe_search = {}
        ssa = getattr(response, "safe_search_annotation", None)
        if ssa:
            try:
                safe_search = {
                    "adult": vision.Likelihood(ssa.adult).name if hasattr(ssa, "adult") else str(ssa.adult),
                    "violence": vision.Likelihood(ssa.violence).name if hasattr(ssa, "violence") else str(ssa.violence),
                    "racy": vision.Likelihood(ssa.racy).name if hasattr(ssa, "racy") else str(ssa.racy),
                }
            except Exception:
                safe_search = {}
        likelihood_map = {
            "UNKNOWN": 0.5,
            "VERY_UNLIKELY": 0.0,
            "UNLIKELY": 0.25,
            "POSSIBLE": 0.5,
            "LIKELY": 0.75,
            "VERY_LIKELY": 1.0,
        }
        safe_search_score = 0.0
        if safe_search:
            vals = [likelihood_map.get(v, 0.5) for v in safe_search.values()]
            safe_search_score = sum(vals) / max(1, len(vals))
        return {"text": text, "labels": labels, "safe_search": safe_search, "safe_search_score": safe_search_score, "raw": {}}
    except Exception as e:
        logger.warning(f"Vision annotate failed: {e}")
        return {"text": "", "labels": [], "safe_search": {}, "safe_search_score": 0.0, "raw": {}}

# Main analyze_content
def analyze_content(text: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    debug: Dict[str, Any] = {}
    parsed = None
    explanation_str = ""
    misp_conf = 0.5

    vision_result = analyze_image_bytes(image_bytes) if image_bytes else {"text": "", "labels": [], "safe_search": {}, "safe_search_score": 0.0, "raw": {}}

    structured_prompt = (
        "You are a fact-check assistant. Given the following input, extract up to 5 factual claims "
        "and return ONLY a JSON object with this schema:\n\n"
        "{\n"
        "  \"claims\": [ {\"text\": \"...\", \"misp_confidence\": 0.0, \"short_reason\": \"...\"} ],\n"
        "  \"overall_misp_confidence\": 0.0,\n"
        "  \"top_reasons\": [\"...\", \"...\"]\n"
        "}\n\n"
        f"Input text:\n{text}\n\nReturn the JSON object only."
    )

    if vertex_available:
        try:
            if _vertex_model_api == "generative_models":
                from vertexai.generative_models import GenerativeModel
                model_id = settings.TEXT_MODEL_ID or "gemini-2.5-flash"
                model = GenerativeModel(model_id)
                response = model.generate_content(structured_prompt)
                raw_text = str(getattr(response, "text", response))
                debug["vertex_raw"] = raw_text[:2000]
                parsed = _vertex_parse_structured(raw_text)
                explanation_str = raw_text
            else:
                from vertexai.language_models import TextGenerationModel
                model_id = settings.TEXT_MODEL_ID or "text-bison@001"
                model = TextGenerationModel.from_pretrained(model_id)
                response = model.predict(structured_prompt, max_output_tokens=512)
                raw_text = str(response) if response else ""
                debug["vertex_raw"] = raw_text[:2000]
                parsed = _vertex_parse_structured(raw_text)
                explanation_str = raw_text
        except Exception as e:
            debug["vertex_error"] = str(e)
            logger.warning(f"Vertex call failed: {e}")
            parsed = None
            explanation_str = f"Vertex call failed: {e}"

    # Normalize parsed confidences if parsed JSON exists to handle inverted numerics
    corrected_overall = None
    if parsed:
        try:
            parsed, corrected_overall = _normalize_parsed_confidences(parsed, explanation_str)
            debug["normalized_overall"] = corrected_overall
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")

    # derive misp_confidence
    if parsed:
        # prefer corrected_overall if available, else fallback to parsed field or numeric in text
        if corrected_overall is not None:
            misp_conf = float(max(0.0, min(1.0, corrected_overall)))
        else:
            overall = parsed.get("overall_misp_confidence")
            if isinstance(overall, (float, int)):
                misp_conf = float(max(0.0, min(1.0, float(overall))))
            else:
                num = _parse_number_from_text(explanation_str)
                misp_conf = float(num) if num is not None else 0.5
    else:
        # fallback heuristics as before
        num = _parse_number_from_text(explanation_str)
        if num is not None:
            misp_conf = float(num)
        else:
            text_lower = text.lower()
            score = 0.0
            suspicious_keywords = ["miracle", "cure", "shocking", "fake", "unverified", "rumor", "scam", "conspiracy", "secret"]
            for kw in suspicious_keywords:
                if kw in text_lower:
                    score += 0.15
            misp_conf = float(max(0.0, min(1.0, score)))
            explanation_str = "FALLBACK HEURISTIC: simple keyword-based detection" if explanation_str == "" else explanation_str

    explanation_pretty = json.dumps(parsed, indent=2) if parsed else explanation_str

    # find candidate sources (best-effort)
    sources_result = []
    try:
        if parsed and isinstance(parsed.get("claims"), list):
            for c in parsed.get("claims", []):
                claim_text = c.get("text") if isinstance(c, dict) else str(c)
                query = f'{claim_text} fact check OR debunk OR "fact-check"'
                hits = _search_web_google_customsearch(query, num=3)
                sources_result.append({"claim": claim_text, "sources": hits})
        else:
            query = f'{text} fact check OR debunk OR "fact-check"'
            hits = _search_web_google_customsearch(query, num=3)
            if hits:
                sources_result.append({"claim": text, "sources": hits})
    except Exception as e:
        logger.warning(f"Source search failed: {e}")

    return {
        "misp_confidence": misp_conf,
        "explanation": explanation_pretty,
        "parsed": parsed,
        "debug": debug,
        "vision": vision_result,
        "sources": sources_result
    }

def map_confidence_to_color(misp_confidence: float) -> str:
    if misp_confidence <= 0.3:
        return "red"
    elif misp_confidence <= 0.6:
        return "orange"
    else:
        return "green"

def send_misinformation_report(subject: str, body: str, to_email: str):
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = settings.SMTP_USER or "no-reply@example.com"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASS)
            server.send_message(msg)
        logger.info(f"Report email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send report email: {e}")
