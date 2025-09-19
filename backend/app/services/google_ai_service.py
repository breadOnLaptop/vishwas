# google_ai_services.py
import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Sequence, cast
from urllib.parse import urlparse

from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings  # your config object

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

# -----------------
# Vision client init
# -----------------
VISION_CLIENT: Optional[Any] = None
vision: Any = None
try:
    from google.cloud import vision as vision_module  # type: ignore
    vision = vision_module
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

# -----------------
# Vertex init
# -----------------
vertex_available: bool = False
_vertex_model_api: Optional[str] = None
try:
    try:
        # try generative_models path
        from vertexai import init as vertex_init  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
        vertex_creds = None
        if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
            vertex_creds = service_account.Credentials.from_service_account_file(
                settings.GOOGLE_APPLICATION_CREDENTIALS
            )
        vertex_init(project=settings.GCP_PROJECT, location=settings.GCP_REGION, credentials=vertex_creds)  # type: ignore
        vertex_available = True
        _vertex_model_api = "generative_models"
        logger.info("Vertex (generative_models) initialized.")
    except Exception as e1:
        try:
            # try language_models path
            from vertexai import init as vertex_init2  # type: ignore
            from vertexai.language_models import TextGenerationModel  # type: ignore
            vertex_creds = None
            if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
                vertex_creds = service_account.Credentials.from_service_account_file(
                    settings.GOOGLE_APPLICATION_CREDENTIALS
                )
            vertex_init2(project=settings.GCP_PROJECT, location=settings.GCP_REGION, credentials=vertex_creds)  # type: ignore
            vertex_available = True
            _vertex_model_api = "language_models"
            logger.info("Vertex (language_models) initialized.")
        except Exception as e2:
            vertex_available = False
            logger.warning(f"Vertex init failed (both paths): {e1} / {e2}")
except Exception as e:
    vertex_available = False
    logger.warning(f"Vertex import/init overall failed: {e}")

# -----------------
# Trusted domains & keywords
# -----------------
_TRUSTED_FACTCHECK_DOMAINS: List[str] = [
    "snopes.com",
    "factcheck.org",
    "politifact.com",
    "apnews.com",
    "reuters.com",
    "bbc.com",
    "fullfact.org",
    "afp.com",
    "factcheck.afp.com",
]
_DEBUNK_KEYWORDS: List[str] = [
    "debunk",
    "false",
    "no evidence",
    "misleading",
    "hoax",
    "not true",
    "debunked",
    "refute",
    "refuted",
]

# -----------------
# Custom Search helper
# -----------------
def _search_web_google_customsearch(
    query: str, num: int = 3, site_filter: Optional[Sequence[str]] = None
) -> List[Dict[str, str]]:
    api_key = settings.GOOGLE_SEARCH_API_KEY
    cx = settings.GOOGLE_SEARCH_CX
    if not api_key or not cx:
        logger.debug("Custom Search API not configured; returning empty sources.")
        return []

    out: List[Dict[str, str]] = []
    try:
        from googleapiclient.discovery import build  # type: ignore

        service = build("customsearch", "v1", developerKey=api_key)
        # If site_filter provided, try site:domain queries first
        if site_filter:
            seen = set()
            for domain in site_filter:
                try:
                    q = f"{query} site:{domain}"
                    res = service.cse().list(q=q, cx=cx, num=min(num, 10)).execute()
                    items = res.get("items", [])[:num]
                    for it in items:
                        link = it.get("link")
                        if not link or link in seen:
                            continue
                        seen.add(link)
                        out.append({"title": it.get("title", ""), "link": link, "snippet": it.get("snippet", "")})
                        if len(out) >= num:
                            return out
                except Exception as exc:
                    logger.debug(f"CustomSearch domain-specific search failed for {domain}: {exc}")

        # generic search or fallback
        if len(out) < num:
            res = service.cse().list(q=query, cx=cx, num=min(num, 10)).execute()
            items = res.get("items", [])[:num]
            for it in items:
                out.append({"title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")})
            return out[:num]
    except Exception as e:
        logger.debug(f"googleapiclient failed: {e}. Trying HTTP fallback.")
        try:
            import httpx  # type: ignore

            params = {"key": api_key, "cx": cx, "q": query, "num": str(num)}
            r = httpx.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                items = data.get("items", [])[:num]
                return [{"title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")} for it in items]
            else:
                logger.warning(f"Custom Search HTTP call failed ({r.status_code}): {r.text}")
        except Exception as e2:
            logger.warning(f"Custom Search HTTP fallback failed: {e2}")
    return out

# -----------------
# small helpers
# -----------------
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


def _strip_code_fences(s: Optional[str]) -> str:
    if not s:
        return ""
    s2 = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I | re.M)
    s2 = re.sub(r"\s*```$", "", s2, flags=re.I | re.M)
    s2 = s2.strip()
    if s2.startswith("`") and s2.endswith("`"):
        s2 = s2.strip("`")
    return s2


def _vertex_parse_structured(text_out: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text_out:
        return None
    txt = _strip_code_fences(text_out)
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = txt[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                candidate_fixed = candidate.replace("'", "\"")
                return json.loads(candidate_fixed)
            except Exception:
                return None
    return None


def _is_trusted_link(link: Optional[str]) -> bool:
    if not link:
        return False
    try:
        host = urlparse(link).netloc.lower()
        for d in _TRUSTED_FACTCHECK_DOMAINS:
            if d in host:
                return True
    except Exception:
        pass
    return False


def _prioritize_sources(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    trusted: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for hit in hits:
        link_var = hit.get("link") or hit.get("url") or ""
        if not link_var or link_var in seen:
            continue
        seen.add(link_var)
        if _is_trusted_link(link_var):
            trusted.append(hit)
        else:
            others.append(hit)
    return trusted + others


def _verify_claim_with_search(claim: str, num: int = 5) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Returns (misp_confidence, hits)
      misp_confidence: 0.0 = debunked / misinformation, 1.0 = strongly supported / truthful.
    Strategy:
      - Search trusted fact-check domains first; if found and snippet contains debunk keywords -> 0.0
      - If trusted article supports claim -> 0.5 (conservative)
      - Otherwise check generic hits and sniff for debunk keywords (0.1) or else uncertain (0.5)
    """
    # trusted domains search
    try:
        trusted_hits = _search_web_google_customsearch(f"{claim} fact check OR debunk", num=num, site_filter=_TRUSTED_FACTCHECK_DOMAINS)
        if trusted_hits:
            hits = _prioritize_sources(trusted_hits)
            for h in hits:
                snippet = (h.get("snippet") or "").lower()
                if any(k in snippet for k in _DEBUNK_KEYWORDS):
                    return 0.0, hits  # debunk found on trusted domain
            # trusted page found but no explicit debunk words -> conservative neutral
            return 0.5, hits
    except Exception as e:
        logger.debug(f"Trusted search failed: {e}")

    # fallback generic search
    try:
        generic_hits = _search_web_google_customsearch(f"{claim} fact check OR debunk", num=num)
        if not generic_hits:
            return 0.5, []
        hits = _prioritize_sources(generic_hits)
        for h in hits:
            snippet = (h.get("snippet") or "").lower()
            if any(k in snippet for k in _DEBUNK_KEYWORDS):
                return 0.1, hits
        return 0.5, hits
    except Exception as e:
        logger.debug(f"Generic search failed: {e}")
    return 0.5, []

# -----------------
# normalize confidences
# -----------------
_truth_keywords: List[str] = [
    "true", "accurate", "accurately", "fundamental", "universally accepted",
    "scientific consensus", "supported by", "evidence", "proven", "confirmed",
    "established", "accepted", "indeed", "correct", "correctly", "accurate",
    "global pandemic", "pandemic", "pandemics", "widespread", "widespreadly",
    "caused", "causes", "cause", "resulted in", "resulted", "led to", "led",
    "significant", "numerous", "health", "economic", "social", "foundational"
]

_false_keywords: List[str] = [
    "no evidence", "debunk", "debunked", "false", "misleading", "fabricated",
    "conspiracy", "untrue", "incorrect", "not true", "disproved", "hoax",
    "fake", "unsupported", "refuted"
]


def _normalize_parsed_confidences(parsed: Dict[str, Any], explanation_text: str) -> Tuple[Dict[str, Any], Optional[float]]:
    if not parsed or not isinstance(parsed, dict):
        return parsed, None

    claims = parsed.get("claims", [])
    corrected_scores: List[float] = []

    context_parts: List[str] = []
    if explanation_text:
        context_parts.append(explanation_text.lower())
    top_reasons = parsed.get("top_reasons", [])
    if isinstance(top_reasons, list):
        context_parts.extend([str(r).lower() for r in top_reasons])
    context_text = " ".join(context_parts)

    for idx, c in enumerate(claims):
        if not isinstance(c, dict):
            c = {"text": str(c)}

        claim_text = (c.get("text") or "").lower()
        short_reason = (c.get("short_reason") or "").lower()
        combined = " ".join([context_text, claim_text, short_reason]).lower()

        t_count = sum(1 for kw in _truth_keywords if kw in combined)
        f_count = sum(1 for kw in _false_keywords if kw in combined)

        textual_inferred_conf: Optional[float] = None
        if t_count > f_count and t_count >= 1:
            textual_inferred_conf = 1.0
        elif f_count > t_count and f_count >= 1:
            textual_inferred_conf = 0.0

        numeric = c.get("misp_confidence", None)
        corrected: float

        if isinstance(numeric, (int, float)):
            numeric = float(numeric)
            if textual_inferred_conf is not None and abs(numeric - textual_inferred_conf) > 0.4:
                corrected = float(textual_inferred_conf)
                logger.debug(
                    f"Normalized claim[{idx}] numeric {numeric} -> {corrected} due to textual cues (t_count={t_count}, f_count={f_count})."
                )
            else:
                corrected = numeric
        else:
            if textual_inferred_conf is not None:
                corrected = float(textual_inferred_conf)
            else:
                corrected = 0.5

        try:
            c["misp_confidence"] = float(corrected)
        except Exception:
            c["misp_confidence"] = corrected

        corrected_scores.append(float(c["misp_confidence"]))

    overall: Optional[float] = None
    if corrected_scores:
        overall = sum(corrected_scores) / len(corrected_scores)
        parsed["overall_misp_confidence"] = overall

    return parsed, overall

# -----------------
# Vision analyze
# -----------------
def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Returns:
      {
        "text": str,
        "labels": List[str],
        "safe_search": Dict[str,str],
        "safe_search_score": Optional[float],
        "web": Dict[str, Any],
        "raw": Dict[str, Any]
      }
    """
    if VISION_CLIENT is None or vision is None:
        return {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "web": {}, "raw": {}}
    try:
        image = vision.Image(content=image_bytes)
        # Build feature array robustly across SDK versions
        features: List[Dict[str, Any]] = []
        try:
            # try the modern descriptor
            features = [
                {"type_": vision.Feature.Type.TEXT_DETECTION},
                {"type_": vision.Feature.Type.LABEL_DETECTION},
                {"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION},
                {"type_": vision.Feature.Type.WEB_DETECTION},
            ]
        except Exception:
            try:
                # older SDK style
                features = [
                    {"type_": vision.enums.Feature.Type.TEXT_DETECTION},
                    {"type_": vision.enums.Feature.Type.LABEL_DETECTION},
                    {"type_": vision.enums.Feature.Type.SAFE_SEARCH_DETECTION},
                    {"type_": vision.enums.Feature.Type.WEB_DETECTION},
                ]
            except Exception:
                # last resort numeric constants (best-effort)
                features = [
                    {"type_": 1},   # TEXT_DETECTION
                    {"type_": 11},  # LABEL_DETECTION
                    {"type_": 9},   # SAFE_SEARCH_DETECTION
                    {"type_": 10},  # WEB_DETECTION
                ]

        response = VISION_CLIENT.annotate_image({
            "image": image,
            "features": features,
        })

        # OCR text
        text_result: str = ""
        if getattr(response, "text_annotations", None):
            try:
                text_result = response.text_annotations[0].description or ""
            except Exception:
                text_result = ""

        # labels
        labels_list: List[str] = [getattr(ann, "description", "") for ann in (getattr(response, "label_annotations", []) or [])]

        # safe search
        safe_search: Dict[str, str] = {}
        ssa = getattr(response, "safe_search_annotation", None)
        if ssa:
            try:
                def _likelihood_name(val: Any) -> str:
                    # attempt multiple variant accesses
                    try:
                        # new style: vision.Likelihood.Name(int(val))
                        if hasattr(vision, "Likelihood") and hasattr(vision.Likelihood, "Name"):
                            return vision.Likelihood.Name(int(val))
                    except Exception:
                        pass
                    try:
                        # object with .name
                        return val.name  # type: ignore
                    except Exception:
                        return str(val)

                safe_search = {
                    "adult": _likelihood_name(getattr(ssa, "adult", None)),
                    "violence": _likelihood_name(getattr(ssa, "violence", None)),
                    "racy": _likelihood_name(getattr(ssa, "racy", None)),
                }
            except Exception:
                safe_search = {}

        likelihood_map: Dict[str, float] = {
            "UNKNOWN": 0.5,
            "VERY_UNLIKELY": 0.0,
            "UNLIKELY": 0.25,
            "POSSIBLE": 0.5,
            "LIKELY": 0.75,
            "VERY_LIKELY": 1.0,
        }
        safe_search_score: Optional[float] = None
        if safe_search:
            try:
                vals = [likelihood_map.get(v, 0.5) for v in safe_search.values()]
                safe_search_score = sum(vals) / max(1, len(vals))
            except Exception:
                safe_search_score = None

        # web detection
        web: Dict[str, Any] = {"entities": [], "pages": [], "full_matching_images": []}
        w = getattr(response, "web_detection", None)
        if w:
            try:
                for e in getattr(w, "web_entities", []) or []:
                    web["entities"].append({
                        "entity_id": getattr(e, "entity_id", None),
                        "score": getattr(e, "score", None),
                        "description": getattr(e, "description", None),
                    })
                for p in getattr(w, "pages_with_matching_images", []) or []:
                    web["pages"].append({
                        "url": getattr(p, "url", None),
                        "page_title": getattr(p, "page_title", None)
                    })
                for fm in getattr(w, "full_matching_images", []) or []:
                    web["full_matching_images"].append({
                        "url": getattr(fm, "url", None),
                        "score": getattr(fm, "score", None)
                    })
            except Exception as e:
                logger.debug(f"web_detection parse issue: {e}")

        return {
            "text": text_result,
            "labels": labels_list,
            "safe_search": safe_search,
            "safe_search_score": safe_search_score,
            "web": web,
            "raw": {}
        }

    except Exception as e:
        logger.warning(f"Vision annotate failed: {e}")
        return {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "web": {}, "raw": {}}


# -----------------
# Main analyze_content
# -----------------
def analyze_content(text: Optional[str], image_bytes: Optional[bytes] = None, source_url: Optional[str] = None) -> Dict[str, Any]:
    debug: Dict[str, Any] = {}
    parsed: Optional[Dict[str, Any]] = None
    explanation_str: str = ""
    misp_conf: float = 0.5

    vision_result = analyze_image_bytes(image_bytes) if image_bytes else {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "web": {}, "raw": {}}
    ocr_text: str = (vision_result.get("text") or "") if isinstance(vision_result, dict) else ""
    image_labels: List[str] = vision_result.get("labels") or []
    web_detection: Dict[str, Any] = cast(Dict[str, Any], vision_result.get("web") or {})
    web_entities: List[str] = [cast(str, e.get("description")) for e in (web_detection.get("entities") or []) if e.get("description")]
    web_pages: List[Dict[str, Any]] = web_detection.get("pages") or []

    # Ignore local file paths for web search
    if source_url and (source_url.startswith("file://") or re.match(r"^[a-zA-Z]:\\", source_url) or source_url.startswith("/")):
        logger.debug("Ignoring local source_url for web lookups.")
        source_url_for_search: Optional[str] = None
    else:
        source_url_for_search = source_url

    if not text:
        text = ocr_text or ""

    # 1) deterministic claims derived from vision/web/OCR
    deterministic_claims: List[Dict[str, Any]] = []
    labelset = [l.lower() for l in image_labels] + [str(e).lower() for e in web_entities if e]

    # barcode detection
    if any("barcode" in s or "bar code" in s for s in labelset):
        deterministic_claims.append({
            "text": "The image shows a barcode printed on or attached to a person's head.",
            "misp_confidence": 0.5,
            "short_reason": "Vision labels or web-detection indicated 'barcode' or similar.",
            "references": []
        })

    # detect human head/back-of-head/bald
    if any(x in " ".join(labelset) for x in ("head", "human", "person", "back of head", "bald", "scalp")):
        deterministic_claims.append({
            "text": "Image depicts the back of a person's head (possibly shaved or bald).",
            "misp_confidence": 1.0,
            "short_reason": "Vision labels or web-detection indicate a human head seen from behind.",
            "references": []
        })

    # OCR claim
    if ocr_text and ocr_text.strip():
        deterministic_claims.append({
            "text": f"Text detected in image or surrounding content: {ocr_text.strip()}",
            "misp_confidence": 0.5,
            "short_reason": "OCR extracted text that may represent a claim.",
            "references": []
        })

    # default hits from vision web pages
    default_page_hits: List[Dict[str, str]] = []
    for p in web_pages:
        url = p.get("url")
        if url:
            default_page_hits.append({"title": p.get("page_title") or url, "link": url, "snippet": ""})

    # If deterministic claims found -> verify them and attach references
    if deterministic_claims:
        parsed = {"claims": deterministic_claims, "overall_misp_confidence": None, "top_reasons": []}
        for c in parsed["claims"]:
            ctext = c.get("text", "")
            conf, hits = _verify_claim_with_search(ctext, num=5)
            if (not hits) and default_page_hits:
                hits = default_page_hits
            c["misp_confidence"] = float(max(0.0, min(1.0, conf)))
            # attach prioritized refs
            c["references"] = _prioritize_sources(hits) if hits else []

        try:
            scores = [float(c.get("misp_confidence", 0.5)) for c in parsed["claims"] if isinstance(c, dict)]
            parsed["overall_misp_confidence"] = (sum(scores) / max(1, len(scores))) if scores else None
            parsed["top_reasons"] = ["Claims derived from vision signals and checked against web/fact-check results."]
            explanation_str = json.dumps(parsed, indent=2)
            debug["deterministic"] = True
        except Exception:
            parsed["overall_misp_confidence"] = None

    # 2) If no deterministic claims, call Vertex once (strict prompt)
    if not parsed:
        strict_prompt = (
            "You are a strict fact-check assistant. ONLY return factual claims that are directly supported "
            "by the provided OCR text, image labels or web-detection. DO NOT INVENT OR GUESS. "
            "Return ONLY a JSON object with keys: claims (array), overall_misp_confidence (0..1), top_reasons (array).\n\n"
            "{\n"
            '  "claims": [ { "text":"...", "misp_confidence":0.0, "short_reason":"...", "references": [] } ],\n'
            '  "overall_misp_confidence": 0.5,\n'
            '  "top_reasons": []\n'
            "}\n\n"
            f"Input OCR text:\n{ocr_text}\n\n"
            f"Image labels: {', '.join(image_labels) if image_labels else '(none)'}\n\n"
            f"Web entities: {', '.join(web_entities) if web_entities else '(none)'}\n\n"
            f"Source URL: {source_url_for_search if source_url_for_search else '(none)'}\n\n"
            "If there are no supported claims, return claims: [] and overall_misp_confidence: 0.5.\n"
            "Return JSON only."
        )

        if vertex_available:
            try:
                model: Any = None
                raw_text: str = ""
                if _vertex_model_api == "generative_models":
                    # GenerativeModel supports generation_config
                    from vertexai.generative_models import GenerativeModel  # type: ignore
                    model_id = settings.TEXT_MODEL_ID or "gemini-2.5-flash"
                    model = GenerativeModel(model_id)
                    # prefer generation_config to set temperature/max tokens
                    try:
                        response = model.generate_content(strict_prompt, generation_config={"temperature": 0.0, "max_output_tokens": 512})
                    except TypeError:
                        # fallback: try without generation_config
                        try:
                            response = model.generate_content(strict_prompt)
                        except Exception as ex:
                            raise ex
                    raw_text = str(getattr(response, "text", response))
                else:
                    # language_models path
                    from vertexai.language_models import TextGenerationModel  # type: ignore
                    model_id = settings.TEXT_MODEL_ID or "text-bison@001"
                    model = TextGenerationModel.from_pretrained(model_id)
                    try:
                        response = model.predict(strict_prompt, max_output_tokens=512, temperature=0.0)
                    except TypeError:
                        response = model.predict(strict_prompt, max_output_tokens=512)
                    raw_text = str(response) if response else ""

                raw_text = _strip_code_fences(raw_text)
                debug["vertex_raw"] = raw_text[:2000]
                parsed_candidate = _vertex_parse_structured(raw_text)
                explanation_str = raw_text
                if parsed_candidate and parsed_candidate.get("claims"):
                    parsed = parsed_candidate
            except Exception as e:
                debug["vertex_error"] = str(e)
                logger.warning(f"Vertex call failed: {e}")
                parsed = None
                explanation_str = f"Vertex call failed: {e}"
        else:
            explanation_str = "Vertex unavailable and no vision signals; unable to create claims."

    # Normalize parsed confidences
    corrected_overall: Optional[float] = None
    if parsed:
        try:
            parsed, corrected_overall = _normalize_parsed_confidences(parsed, explanation_str)
            debug["normalized_overall"] = corrected_overall
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")

    # derive misp_confidence (0..1 where 0 = misinformation, 1 = truthful)
    if parsed:
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
        num = _parse_number_from_text(explanation_str)
        if num is not None:
            misp_conf = float(num)
        else:
            text_lower = (text or "").lower()
            score = 0.0
            suspicious_keywords = ["miracle", "cure", "shocking", "fake", "unverified", "rumor", "scam", "conspiracy", "secret"]
            for kw in suspicious_keywords:
                if kw in text_lower:
                    score += 0.15
            misp_conf = float(max(0.0, min(1.0, score)))
            explanation_str = "FALLBACK HEURISTIC: simple keyword-based detection" if explanation_str == "" else explanation_str

    explanation_pretty = json.dumps(parsed, indent=2) if parsed else explanation_str

    # find candidate sources for each claim - prefer Vision web pages, then model refs, then customsearch
    sources_result: List[Dict[str, Any]] = []
    top_links: List[Dict[str, Any]] = []
    try:
        vision_pages = web_detection.get("pages") or []

        if parsed and isinstance(parsed.get("claims"), list):
            for c in parsed.get("claims", []):
                claim_text = c.get("text") if isinstance(c, dict) else str(c)
                # model provided references?
                existing_refs = c.get("references") if isinstance(c, dict) and c.get("references") else []

                # attach vision pages if none
                if not existing_refs and vision_pages:
                    vis_hits: List[Dict[str, Any]] = []
                    for p in vision_pages:
                        url = p.get("url")
                        if not url:
                            continue
                        vis_hits.append({"title": p.get("page_title") or url, "link": url, "snippet": ""})
                    if vis_hits:
                        existing_refs = vis_hits

                # still none -> custom search
                if not existing_refs:
                    query = f'{claim_text} fact check OR debunk OR "fact-check"'
                    hits = _search_web_google_customsearch(query, num=3)
                    existing_refs = hits

                try:
                    c["references"] = existing_refs
                except Exception:
                    pass

                sources_result.append({"claim": claim_text, "sources": existing_refs})
                for hit in existing_refs:
                    link_var = hit.get("link")
                    if link_var and link_var not in [t.get("link") for t in top_links]:
                        top_links.append(hit)
        else:
            query_parts: List[str] = []
            if text:
                query_parts.append(text)
            if image_labels:
                query_parts.append(" ".join(image_labels))
            if source_url_for_search:
                query_parts.append(source_url_for_search)
            if vision_pages:
                for p in vision_pages:
                    url = p.get("url")
                    if url:
                        top_links.append({"title": p.get("page_title") or url, "link": url, "snippet": ""})

            query = " ".join(query_parts).strip()
            if query:
                hits = _search_web_google_customsearch(query, num=5)
                if hits:
                    sources_result.append({"claim": query, "sources": hits})
                    for hit in hits:
                        link_var = hit.get("link")
                        if link_var and link_var not in [t.get("link") for t in top_links]:
                            top_links.append(hit)
    except Exception as e:
        logger.warning(f"Source search failed: {e}")

    # -----------------
    # Build user-facing explanation (summary)
    # -----------------
    user_explanation_parts: List[str] = []

    # Interpret score
    if misp_conf <= 0.3:
        user_explanation_parts.append("⚠️ The content is likely misinformation based on fact-check evidence and signals.")
    elif misp_conf <= 0.6:
        user_explanation_parts.append("❓ The content shows signs of uncertainty or mixed signals — treat with caution.")
    else:
        user_explanation_parts.append("✅ The content appears credible with no strong misinformation signals detected.")

    # Mention OCR / text evidence
    if ocr_text:
        snippet = ocr_text.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200].strip() + "..."
        user_explanation_parts.append(f"OCR detected text: \"{snippet}\"")

    # Mention claims
    if parsed and parsed.get("claims"):
        claim_count = len(parsed["claims"])
        user_explanation_parts.append(f"{claim_count} claim(s) were extracted and checked against web/fact-check sources.")
        # summarize up to 3 claims
        for c in parsed["claims"][:3]:
            ct = (c.get("text") or "").replace("\n", " ").strip()
            if len(ct) > 160:
                ct = ct[:160].strip() + "..."
            user_explanation_parts.append(f"• \"{ct}\" — confidence: {float(c.get('misp_confidence', 0.5)):.2f}")
    else:
        user_explanation_parts.append("No explicit claims were confidently extracted from this content.")

    # Mention references if any
    if top_links:
        user_explanation_parts.append(f"Fact-check / source links were identified ({len(top_links)}).")
    else:
        user_explanation_parts.append("No direct fact-check references were located.")

    user_explanation = " ".join(user_explanation_parts)

    return {
        "misp_confidence": misp_conf,
        "color": map_confidence_to_color(misp_conf),
        "explanation": explanation_pretty,
        "user_explanation": user_explanation,
        "parsed": parsed,
        "debug": debug,
        "vision": vision_result,
        "sources": sources_result,
        "top_sources": top_links
    }


# map to color (small helper)
def map_confidence_to_color(misp_confidence: float) -> str:
    """
    misp_confidence: 0..1 where 0=mispinfo, 1=truthful.
    Lower -> red, mid -> orange, high -> green
    """
    try:
        mc = float(misp_confidence)
    except Exception:
        mc = 0.5
    if mc <= 0.3:
        return "red"
    elif mc <= 0.6:
        return "orange"
    else:
        return "green"


# send report (safe-guard optional SMTP values)
def send_misinformation_report(subject: str, body: str, to_email: str) -> None:
    smtp_host = settings.SMTP_HOST or ""
    smtp_port = settings.SMTP_PORT or 0
    smtp_user = settings.SMTP_USER or ""
    smtp_pass = settings.SMTP_PASS or ""

    if not smtp_host or not smtp_port:
        logger.error("SMTP host/port not configured; cannot send email.")
        return

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = smtp_user or "peeyushmaurya.dev@gmail.com"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"Report email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send report email: {e}")
