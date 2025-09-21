import os
import re
import json
import logging
import textwrap
import math
from typing import Dict, Any, Optional, List, Tuple, Sequence, cast
from urllib.parse import urlparse
from datetime import datetime
 
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
# Vision client init (robust)
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
# Vertex init (try both generative_models and language_models)
# -----------------
vertex_available: bool = False
_vertex_model_api: Optional[str] = None
try:
    try:
        from vertexai import init as vertex_init  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
        vertex_creds = None
        if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
            vertex_creds = service_account.Credentials.from_service_account_file(settings.GOOGLE_APPLICATION_CREDENTIALS)
        vertex_init(project=settings.GCP_PROJECT, location=settings.GCP_REGION, credentials=vertex_creds)  # type: ignore
        vertex_available = True
        _vertex_model_api = "generative_models"
        logger.info("Vertex (generative_models) initialized.")
    except Exception as e1:
        try:
            from vertexai import init as vertex_init2  # type: ignore
            from vertexai.language_models import TextGenerationModel  # type: ignore
            vertex_creds = None
            if settings.GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
                vertex_creds = service_account.Credentials.from_service_account_file(settings.GOOGLE_APPLICATION_CREDENTIALS)
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
# Trusted domains & authoritative support domains
# -----------------
_TRUSTED_FACTCHECK_DOMAINS: List[str] = [
    "snopes.com", "factcheck.org", "politifact.com", "apnews.com",
    "reuters.com", "bbc.com", "fullfact.org", "afp.com", "factcheck.afp.com"
]
_TRUSTED_SUPPORT_DOMAINS: List[str] = [
    "nasa.gov", "noaa.gov", "who.int", "cdc.gov", "nih.gov", "nhs.uk",
    "clevelandclinic.org", "mayoclinic.org", "bhf.org.uk", "webmd.com", "healthline.com"
]


# -----------------
# Decision thresholds (tunable)
# -----------------
VERTEX_UNIVERSAL_THRESHOLD = 0.90  # vertex must be >= this to accept universal fact
VERTEX_TRUE_THRESHOLD = 0.90
VERTEX_FALSE_THRESHOLD = 0.10
SIMILARITY_THRESHOLD = 0.7
SUPPORT_SIMILARITY_THRESHOLD = 0.45
SUPPORT_CHECK_UPPER = 0.85


# -----------------
# Helper: normalize a reference / source dict
# -----------------
def _normalize_ref(r: Dict[str, Any]) -> Dict[str, str]:
    try:
        title = r.get("title") or r.get("publisher") or ""
        link = r.get("link") or r.get("claim_url") or r.get("url") or ""
        snippet = r.get("snippet") or r.get("text") or ""
        publisher = r.get("publisher") or ""
        # sanitize lengths to avoid huge payloads
        return {
            "title": str(title)[:300],
            "link": str(link)[:2000],
            "snippet": str(snippet)[:800],
            "publisher": str(publisher)[:200],
        }
    except Exception:
        return {"title": "", "link": "", "snippet": "", "publisher": ""}


# -----------------
# Custom Search helper
# -----------------
def _search_web_google_customsearch(query: str, num: int = 3, site_filter: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    api_key = settings.GOOGLE_SEARCH_API_KEY
    cx = settings.GOOGLE_SEARCH_CX
    if not api_key or not cx:
        logger.debug("Custom Search API not configured; returning empty sources.")
        return []

    out: List[Dict[str, str]] = []
    try:
        from googleapiclient.discovery import build  # type: ignore
        service = build("customsearch", "v1", developerKey=api_key)

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

        if len(out) < num:
            res = service.cse().list(q=query, cx=cx, num=min(num, 10)).execute()
            items = res.get("items", [])[:num]
            for it in items:
                out.append({"title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")})
            return out[:num]
    except Exception as e:
        logger.debug(f"googleapiclient failed: {e}. Using HTTP fallback.")
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
# Fact Check Tools helper (HTTP call)
# -----------------
def _factcheck_search(claim: str, num: int = 5) -> List[Dict[str, Any]]:
    api_key = settings.GOOGLE_SEARCH_API_KEY
    if not api_key:
        return []

    try:
        import httpx  # type: ignore
        base = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": claim, "pageSize": str(num), "key": api_key}
        r = httpx.get(base, params=params, timeout=8.0)
        if r.status_code != 200:
            logger.debug(f"Fact Check API HTTP {r.status_code}: {r.text}")
            return []
        data = r.json()
        results: List[Dict[str, Any]] = []
        for it in data.get("claims", [])[:num]:
            for cr in it.get("claimReview", []):
                results.append({
                    "text": it.get("text"),
                    "claim_url": cr.get("url"),
                    "publisher": (cr.get("publisher") or {}).get("name"),
                    "textualRating": cr.get("textualRating"),
                    "title": cr.get("title"),
                    "snippet": cr.get("title") or cr.get("text") or ""
                })
        return results
    except Exception as e:
        logger.debug(f"FactCheck API call failed: {e}")
        return []


# -----------------
# Additional helpers & normalization
# -----------------
def _normalize_textual_rating_to_verdict(textual: Optional[str]) -> Optional[str]:
    if not textual:
        return None
    t = str(textual).lower()
    false_keys = ["false", "pants on fire", "incorrect", "not true", "hoax", "debunk", "refuted", "refute"]
    true_keys = ["true", "correct", "supports", "confirmed", "accurate"]
    for k in false_keys:
        if k in t:
            return "false"
    for k in true_keys:
        if k in t:
            return "true"
    if "mostly true" in t or "partly" in t or "mixture" in t:
        return "mixed"
    return None


def _pick_decisive_factcheck(fc_hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not fc_hits:
        return None
    for h in fc_hits:
        v = _normalize_textual_rating_to_verdict(h.get("textualRating") or h.get("snippet") or "")
        if v == "false":
            h_copy = dict(h)
            h_copy["verdict"] = "false"
            return h_copy
    for h in fc_hits:
        v = _normalize_textual_rating_to_verdict(h.get("textualRating") or h.get("snippet") or "")
        if v == "true":
            h_copy = dict(h)
            h_copy["verdict"] = "true"
            return h_copy
    h_copy = dict(fc_hits[0])
    h_copy["verdict"] = "undecided"
    return h_copy


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


def _is_supportive_domain(link: Optional[str]) -> bool:
    if not link:
        return False
    try:
        host = urlparse(link).netloc.lower()
        for d in _TRUSTED_SUPPORT_DOMAINS:
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
        link_var = hit.get("link") or hit.get("claim_url") or hit.get("url") or ""
        if not link_var or link_var in seen:
            continue
        seen.add(link_var)
        if _is_trusted_link(link_var):
            trusted.append(hit)
        else:
            others.append(hit)
    combined = trusted + others
    # normalize each ref
    return [_normalize_ref(dict(h)) for h in combined]


# -----------------
# semantic similarity (fallback)
# -----------------
def semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set([t for t in re.findall(r"\w+", a.lower()) if len(t) > 2])
    tb = set([t for t in re.findall(r"\w+", b.lower()) if len(t) > 2])
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    union = ta.union(tb)
    return float(len(inter)) / float(len(union))


# -----------------
# Vertex: classification helper (used for general classification)
# -----------------
def vertex_classify_claim(claim_text: str, context_snippets: List[Dict[str, str]], model: str = None) -> Dict[str, Any]:
    model = model or (settings.TEXT_MODEL_ID or ("gemini-2.5-flash" if _vertex_model_api == "generative_models" else "text-bison@001"))

    ctx_pieces = []
    for s in (context_snippets or [])[:6]:
        t = (s.get("title") or "")[:240].strip()
        sn = (s.get("snippet") or "")[:400].strip()
        url = s.get("link") or s.get("url") or ""
        ctx_pieces.append(f"TITLE: {t}\nSNIPPET: {sn}\nURL: {url}")
    context_text = "\n\n".join(ctx_pieces)

    prompt = (
        "You are a precise verifier. Input: a CLAIM and up to 6 CONTEXT snippets. "
        "Output: a single JSON object (no extra text) with keys exactly: "
        "verdict (true|false|mixed|undecided), misp_confidence (0.0-1.0), misp_confidence_0_1 (same), "
        "confidence_0_10 (0-10), explanation (short string), references (array of objects with title,url,snippet,verdict).\n\n"
        f"CLAIM: {claim_text}\n\nCONTEXT:\n{context_text}\n\nJSON:"
    )

    try:
        raw = ""
        if vertex_available:
            if _vertex_model_api == "generative_models":
                from vertexai.generative_models import GenerativeModel  # type: ignore
                m = GenerativeModel(model)
                try:
                    response = m.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 600})
                except TypeError:
                    response = m.generate_content(prompt)
                raw = str(getattr(response, "text", response))
            else:
                from vertexai.language_models import TextGenerationModel  # type: ignore
                m = TextGenerationModel.from_pretrained(model)
                try:
                    response = m.predict(prompt, max_output_tokens=600, temperature=0.0)
                except TypeError:
                    response = m.predict(prompt, max_output_tokens=600)
                raw = str(response) if response else ""
        else:
            raise NotImplementedError("Vertex not available")
        raw_text = _strip_code_fences(raw)
        parsed = _vertex_parse_structured(raw_text)
    except NotImplementedError:
        parsed = None
    except Exception as e:
        logger.debug(f"vertex_classify_claim exception: {e}")
        parsed = None

    if not parsed:
        return {
            "verdict": "undecided",
            "misp_confidence": 0.5,
            "misp_confidence_0_1": 0.5,
            "confidence_0_10": 5.0,
            "explanation": "Vertex unavailable or returned invalid JSON; defaulted to uncertain.",
            "references": context_snippets or []
        }

    try:
        mc = float(parsed.get("misp_confidence", parsed.get("misp_confidence_0_1", 0.5)))
    except Exception:
        mc = 0.5
    mc = max(0.0, min(1.0, mc))
    try:
        c10 = float(parsed.get("confidence_0_10", mc * 10.0))
    except Exception:
        c10 = mc * 10.0
    c10 = max(0.0, min(10.0, c10))

    parsed["misp_confidence"] = mc
    parsed["misp_confidence_0_1"] = mc
    parsed["confidence_0_10"] = c10
    parsed["verdict"] = str(parsed.get("verdict") or "").lower() or None

    refs = parsed.get("references") or []
    normalized_refs = []
    for r in refs:
        try:
            normalized_refs.append({
                "title": r.get("title", "") if isinstance(r, dict) else str(r)[:200],
                "link": r.get("url", r.get("link", "")) if isinstance(r, dict) else "",
                "snippet": r.get("snippet", "") if isinstance(r, dict) else "",
                "verdict": r.get("verdict", "") if isinstance(r, dict) else ""
            })
        except Exception:
            continue
    if not normalized_refs:
        normalized_refs = context_snippets or []

    parsed["references"] = normalized_refs
    parsed["explanation"] = parsed.get("explanation", "") or ""
    return parsed


# -----------------
# NEW: Vertex-only universal check
# -----------------
def vertex_universal_check(claim_text: str, model: str = None) -> Dict[str, Any]:
    """
    Ask Vertex if the claim is a universal/truth-of-fact statement.
    Only Vertex is used here (no keyword heuristics). Vertex must return JSON with:
      { "is_universal": true|false, "misp_confidence": 0.0-1.0, "explanation": "..."}
    If Vertex fails to return JSON, fallback to {is_universal: False, misp_confidence: 0.5}
    """
    model = model or (settings.TEXT_MODEL_ID or ("gemini-2.5-flash" if _vertex_model_api == "generative_models" else "text-bison@001"))

    prompt = (
        "You are a strict factuality checker. Answer ONLY a JSON object (no extra commentary). "
        "Task: given a short factual sentence (CLAIM), decide whether it should be considered a "
        "'universal factual statement' (i.e., widely accepted, context-free factual statement like "
        "'The Earth orbits the Sun', 'Water freezes at 0 °C at sea level', etc.).\n\n"
        "OUTPUT SCHEMA (required):\n"
        "{\n"
        '  "is_universal": true|false,\n'
        '  "misp_confidence": 0.0, # numeric 0..1 representing how confident you are that this is universally true\n'
        '  "explanation": "one-line explanation or empty"\n'
        "}\n\n"
        f"CLAIM: {claim_text}\n\nJSON:"
    )

    try:
        raw = ""
        if vertex_available:
            if _vertex_model_api == "generative_models":
                from vertexai.generative_models import GenerativeModel  # type: ignore
                m = GenerativeModel(model)
                try:
                    response = m.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 300})
                except TypeError:
                    response = m.generate_content(prompt)
                raw = str(getattr(response, "text", response))
            else:
                from vertexai.language_models import TextGenerationModel  # type: ignore
                m = TextGenerationModel.from_pretrained(model)
                try:
                    response = m.predict(prompt, max_output_tokens=300, temperature=0.0)
                except TypeError:
                    response = m.predict(prompt, max_output_tokens=300)
                raw = str(response) if response else ""
        else:
            raise NotImplementedError("Vertex not available")
        raw_text = _strip_code_fences(raw)
        parsed = _vertex_parse_structured(raw_text)
    except Exception as e:
        logger.debug(f"vertex_universal_check failed: {e}")
        parsed = None

    if not parsed:
        return {"is_universal": False, "misp_confidence": 0.5, "explanation": "Vertex unavailable or returned invalid JSON."}

    is_univ = bool(parsed.get("is_universal", False))
    try:
        mc = float(parsed.get("misp_confidence", 0.5))
    except Exception:
        mc = 0.5
    mc = max(0.0, min(1.0, mc))
    return {"is_universal": is_univ, "misp_confidence": mc, "explanation": parsed.get("explanation", ""), "raw": parsed}


# -----------------
# verify_claim_pipeline (updated sequence as requested)
# -----------------
def verify_claim_pipeline(claim_obj: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
    """
    Sequence:
      1) Vertex-only universal check (if vertex says universal with high confidence -> accept 10/10).
      2) FactCheck API lookup (if decisive -> use its verdict).
      3) CustomSearch (trusted support domains) -> if supportive authoritative page found -> accept 10/10.
      4) Final Vertex call -> use vertex numeric score directly as final misp_confidence.
    """
    claim_text = (
        claim_obj.get("claim_text")
        or claim_obj.get("text")
        or claim_obj.get("claim")
        or claim_obj.get("title")
        or ""
    )
    claim_text = str(claim_text).strip()
    claim_obj.setdefault("references", [])
    claim_obj.setdefault("explanation", "")

    # 0) keep any pre-existing numeric
    existing_num = None
    try:
        if isinstance(claim_obj.get("misp_confidence"), (int, float, str)):
            existing_num = float(claim_obj.get("misp_confidence"))
    except Exception:
        existing_num = None

    # --- STEP 1: Vertex-only universal check
    try:
        uni = vertex_universal_check(claim_text, model=(settings.TEXT_MODEL_ID or None))
        logger.debug(f"vertex_universal_check -> {uni}")
    except Exception as e:
        logger.debug(f"vertex_universal_check exception: {e}")
        uni = {"is_universal": False, "misp_confidence": 0.5, "explanation": "vertex check error"}

    if uni.get("is_universal") and float(uni.get("misp_confidence", 0.5)) >= VERTEX_UNIVERSAL_THRESHOLD:
        claim_obj["misp_confidence"] = 1.0
        claim_obj["misp_confidence_0_1"] = 1.0
        claim_obj["confidence_0_10"] = 10.0
        claim_obj["explanation"] = f"Vertex universal-check accepted claim (confidence={uni.get('misp_confidence')})."
        claim_obj["references"] = [] if not claim_obj.get("references") else claim_obj.get("references")
        claim_obj["overridden_by_universal_vertex"] = True
        return claim_obj

    # --- STEP 2: FactCheck API lookup (authoritative)
    try:
        fc_hits = _factcheck_search(claim_text, num=6)
        if fc_hits:
            decisive = _pick_decisive_factcheck(fc_hits)
            if decisive:
                textual_rating = decisive.get("textualRating") or decisive.get("snippet") or ""
                mapped = {"misp_confidence": 0.5, "misp_confidence_0_1": 0.5, "confidence_0_10": 5.0}
                # map textual to scores
                if _normalize_textual_rating_to_verdict(textual_rating) == "false":
                    mapped = {"misp_confidence": 0.0, "misp_confidence_0_1": 0.0, "confidence_0_10": 0.0}
                elif _normalize_textual_rating_to_verdict(textual_rating) == "true":
                    mapped = {"misp_confidence": 1.0, "misp_confidence_0_1": 1.0, "confidence_0_10": 10.0}
                else:
                    mapped = {"misp_confidence": 0.5, "misp_confidence_0_1": 0.5, "confidence_0_10": 5.0}

                claim_obj["misp_confidence"] = mapped["misp_confidence"]
                claim_obj["misp_confidence_0_1"] = mapped["misp_confidence_0_1"]
                claim_obj["confidence_0_10"] = mapped["confidence_0_10"]
                claim_obj["explanation"] = f"Authoritative fact-check matched (publisher={decisive.get('publisher')})."
                claim_obj["references"] = [{
                    "title": decisive.get("title") or decisive.get("text") or "",
                    "link": decisive.get("claim_url") or decisive.get("url") or "",
                    "snippet": decisive.get("snippet") or "",
                    "publisher": decisive.get("publisher") or ""
                }]
                claim_obj["overridden_by_factcheck"] = True
                return claim_obj
    except Exception as e:
        logger.debug(f"Fact-check lookup failed: {e}")

    # --- STEP 3: CustomSearch for trusted support domains (if found -> accept)
    try:
        support_hits = _search_web_google_customsearch(claim_text, num=5, site_filter=_TRUSTED_SUPPORT_DOMAINS) or []
        if support_hits:
            # pick first relevant supportive hit (no fuzzy scoring here; presence is treated as supportive)
            first = support_hits[0]
            if first and first.get("link"):
                # Mark as supported by authoritative domain => accept true
                claim_obj["misp_confidence"] = 1.0
                claim_obj["misp_confidence_0_1"] = 1.0
                claim_obj["confidence_0_10"] = 10.0
                claim_obj["explanation"] = f"Supportive authoritative page found: {first.get('link')}"
                claim_obj["references"] = [_normalize_ref(first)]
                claim_obj["overridden_by_support_search"] = True
                return claim_obj
    except Exception as e:
        logger.debug(f"Supportive customsearch failed: {e}")

    # --- STEP 4: Final Vertex judgement (use Vertex numeric score directly as final)
    try:
        # gather some context from general customsearch (not used to override Vertex scoring — Vertex decides)
        contexts = _search_web_google_customsearch(claim_text, num=6) or []
    except Exception:
        contexts = []

    try:
        vertex_final = vertex_classify_claim(claim_text, contexts, model=(settings.TEXT_MODEL_ID or None))
        logger.debug(f"vertex_final -> {vertex_final}")
    except Exception as e:
        logger.debug(f"vertex_classify_claim failed: {e}")
        vertex_final = {"misp_confidence": 0.5, "verdict": "undecided", "explanation": "vertex error", "references": contexts}

    final_mc = float(max(0.0, min(1.0, float(vertex_final.get("misp_confidence", 0.5)))))
    claim_obj["misp_confidence"] = final_mc
    claim_obj["misp_confidence_0_1"] = final_mc
    claim_obj["confidence_0_10"] = round(final_mc * 10.0, 2)
    claim_obj["explanation"] = vertex_final.get("explanation", "") or claim_obj.get("explanation", "")
    claim_obj["references"] = vertex_final.get("references", claim_obj.get("references", []))
    claim_obj["vertex_final_raw"] = vertex_final
    # no additional combination — use vertex numeric directly
    return claim_obj


# -----------------
# normalize confidences (uses verify pipeline)
# -----------------
def _normalize_parsed_confidences(parsed: Dict[str, Any], explanation_text: str) -> Tuple[Dict[str, Any], Optional[float]]:
    if not parsed or not isinstance(parsed, dict):
        return parsed, None

    claims = parsed.get("claims", [])
    corrected_scores: List[float] = []

    for idx, c in enumerate(claims):
        if not isinstance(c, dict):
            c = {"text": str(c)}
            claims[idx] = c

        try:
            updated = verify_claim_pipeline(c)
            claims[idx] = updated
            corrected_scores.append(float(updated.get("misp_confidence", 0.5)))
        except Exception as e:
            logger.debug(f"_normalize_parsed_confidences: verify_claim_pipeline failed for claim '{c.get('text','')}' : {e}")
            try:
                mc = float(c.get("misp_confidence", 0.5))
            except Exception:
                mc = 0.5
            c["misp_confidence"] = mc
            corrected_scores.append(mc)

    overall: Optional[float] = None
    if corrected_scores:
        overall = sum(corrected_scores) / len(corrected_scores)
        parsed["overall_misp_confidence"] = overall

    return parsed, overall


# -----------------
# Vision analyze (unchanged)
# -----------------
def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    if VISION_CLIENT is None or vision is None:
        return {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "web": {}, "raw": {}}
    try:
        image = vision.Image(content=image_bytes)
        features: List[Dict[str, Any]] = []
        try:
            features = [
                {"type_": vision.Feature.Type.TEXT_DETECTION},
                {"type_": vision.Feature.Type.LABEL_DETECTION},
                {"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION},
                {"type_": vision.Feature.Type.WEB_DETECTION},
            ]
        except Exception:
            try:
                features = [
                    {"type_": vision.enums.Feature.Type.TEXT_DETECTION},
                    {"type_": vision.enums.Feature.Type.LABEL_DETECTION},
                    {"type_": vision.enums.Feature.Type.SAFE_SEARCH_DETECTION},
                    {"type_": vision.enums.Feature.Type.WEB_DETECTION},
                ]
            except Exception:
                features = [{"type_": 1}, {"type_": 11}, {"type_": 9}, {"type_": 10}]

        response = VISION_CLIENT.annotate_image({"image": image, "features": features})

        text_result: str = ""
        if getattr(response, "text_annotations", None):
            try:
                text_result = response.text_annotations[0].description or ""
            except Exception:
                text_result = ""

        labels_list: List[str] = [getattr(ann, "description", "") for ann in (getattr(response, "label_annotations", []) or [])]

        safe_search: Dict[str, str] = {}
        ssa = getattr(response, "safe_search_annotation", None)
        if ssa:
            try:
                def _likelihood_name(val: Any) -> str:
                    try:
                        if hasattr(vision, "Likelihood") and hasattr(vision.Likelihood, "Name"):
                            return vision.Likelihood.Name(int(val))
                    except Exception:
                        pass
                    try:
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
            "UNKNOWN": 0.5, "VERY_UNLIKELY": 0.0, "UNLIKELY": 0.25,
            "POSSIBLE": 0.5, "LIKELY": 0.75, "VERY_LIKELY": 1.0,
        }
        safe_search_score: Optional[float] = None
        if safe_search:
            try:
                vals = [likelihood_map.get(v, 0.5) for v in safe_search.values()]
                safe_search_score = sum(vals) / max(1, len(vals))
            except Exception:
                safe_search_score = None

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

        return {"text": text_result, "labels": labels_list, "safe_search": safe_search, "safe_search_score": safe_search_score, "web": web, "raw": {}}

    except Exception as e:
        logger.warning(f"Vision annotate failed: {e}")
        return {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "web": {}, "raw": {}}


# -----------------
# Main analyze_content (uses verify_claim_pipeline)
# -----------------
def analyze_content(text: Optional[str], image_bytes: Optional[bytes] = None, source_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze text + optional image.

    Returns canonical dict with:
      - score: 0..10 (trustworthiness; higher = more truthful)
      - color: "red"|"orange"|"green"
      - user_explanation: short summary
      - top_sources: normalized list
      - parsed: structured claims with per-claim confidence
      - debug: internal debug info
    """
    debug: Dict[str, Any] = {}
    parsed: Optional[Dict[str, Any]] = None
    explanation_str: str = ""
    misp_conf: float = 0.5

    # call vision if image bytes given
    vision_result = analyze_image_bytes(image_bytes) if image_bytes else {"text": "", "labels": [], "safe_search": {}, "safe_search_score": None, "web": {}, "raw": {}}
    ocr_text: str = (vision_result.get("text") or "") if isinstance(vision_result, dict) else ""
    image_labels: List[str] = vision_result.get("labels") or []
    web_detection: Dict[str, Any] = cast(Dict[str, Any], vision_result.get("web") or {})
    web_entities: List[str] = [cast(str, e.get("description")) for e in (web_detection.get("entities") or []) if e.get("description")]
    web_pages: List[Dict[str, Any]] = web_detection.get("pages") or []

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

    # If deterministic claims found -> verify them (new sequence)
    if deterministic_claims:
        parsed = {"claims": deterministic_claims, "overall_misp_confidence": None, "top_reasons": []}
        for i, c in enumerate(parsed["claims"]):
            try:
                updated = verify_claim_pipeline(c)
            except Exception as e:
                logger.debug(f"deterministic verify_claim_pipeline failure: {e}")
                updated = c
            if (not updated.get("references")) and default_page_hits:
                updated["references"] = default_page_hits
            parsed["claims"][i] = updated
        try:
            scores = [float(c.get("misp_confidence", 0.5)) for c in parsed["claims"] if isinstance(c, dict)]
            parsed["overall_misp_confidence"] = (sum(scores) / max(1, len(scores))) if scores else None
            parsed["top_reasons"] = ["Claims derived from vision signals were checked via Vertex (+fact-check/support if needed)."]
            explanation_str = json.dumps(parsed, indent=2)
            debug["deterministic"] = True
        except Exception:
            parsed["overall_misp_confidence"] = None

    # 2) If no deterministic claims, call Vertex once (strict prompt) -> then verify each claim using sequence
    if not parsed:
        strict_prompt = (
            "You are a strict factual-evidence assistant used inside an automated pipeline. "
            "OBEY THESE INSTRUCTIONS EXACTLY and return ONLY a JSON object (no natural-language explanation, no extra keys):\n\n"

            "OUTPUT SCHEMA (required):\n"
            "{\n"
            '  "claims": [\n'
            '    {\n'
            '      "text": "<claim text>",\n'
            '      "verdict": "<true|false|mixed|undecided>",\n'
            '      "misp_confidence": 0.0,  # number between 0.0 (false/misinformation) and 1.0 (true/accurate)\n'
            '      "short_reason": "<one-line reason why verdict was chosen>",\n'
            '      "evidence": [ { "title":"", "link":"", "publisher":"", "snippet":"", "authoritative": true|false } ]\n'
            '    }\n'
            '  ],\n'
            '  "overall_misp_confidence": 0.5,  # numeric 0..1 (average truthiness across claims)\n'
            '  "top_reasons": [ "<short human-readable bullets why claims were rated>" ]\n'
            "}\n\n"

            "MANDATORY RULES (must follow):\n"
            "1) IDENTIFY ONLY factual claims that are directly supported by the INPUT (OCR text, image labels, web-entities, or Source URL). DO NOT INVENT claims.\n"
            "2) For each claim set `verdict` to exactly one of: true, false, mixed, undecided.\n"
            "3) `misp_confidence` MUST be consistent with `verdict` using mapping (false->0.0,true->1.0,mixed/undecided->0.5).\n"
            "4) Populate `evidence` entries. Mark `authoritative=true` for known fact-checkers or scientific organizations.\n"
            "5) If no claims can be extracted, return `claims: []` and `overall_misp_confidence: 0.5`.\n\n"

            "INPUT (use these when extracting claims):\n"
            f"Input OCR text:\n{ocr_text}\n\n"
            f"Image labels: {', '.join(image_labels) if image_labels else '(none)'}\n\n"
            f"Web entities: {', '.join(web_entities) if web_entities else '(none)'}\n\n"
            f"Source URL: {source_url_for_search if source_url_for_search else '(none)'}\n\n"

            "IMPORTANT: JSON ONLY. No commentary, no extra keys outside the schema."
        )

        if vertex_available:
            try:
                model: Any = None
                raw_text: str = ""
                if _vertex_model_api == "generative_models":
                    from vertexai.generative_models import GenerativeModel  # type: ignore
                    model_id = settings.TEXT_MODEL_ID or "gemini-2.5-flash"
                    model = GenerativeModel(model_id)
                    try:
                        response = model.generate_content(strict_prompt, generation_config={"temperature": 0.0, "max_output_tokens": 512})
                    except TypeError:
                        response = model.generate_content(strict_prompt)
                    raw_text = str(getattr(response, "text", response))
                else:
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
                    normalized_claims = []
                    for c in parsed_candidate.get("claims", []):
                        claim_dict = dict(c) if isinstance(c, dict) else {"text": str(c)}
                        if claim_dict.get("evidence") and not claim_dict.get("references"):
                            claim_dict["references"] = claim_dict.get("evidence")
                        try:
                            verified = verify_claim_pipeline(claim_dict)
                        except Exception as e:
                            logger.debug(f"verify_claim_pipeline failed on model claim: {e}")
                            verified = claim_dict
                        normalized_claims.append(verified)
                    parsed = {"claims": normalized_claims, "overall_misp_confidence": None, "top_reasons": parsed_candidate.get("top_reasons", [])}
            except Exception as e:
                debug["vertex_error"] = str(e)
                logger.warning(f"Vertex call failed: {e}")
                parsed = None
                explanation_str = f"Vertex call failed: {e}"
        else:
            explanation_str = "Vertex unavailable and no vision signals; unable to create claims."

    # --- DIRECT fallback on the text if still nothing parsed
    if not parsed and text and text.strip():
        try:
            claim_obj = {"text": text.strip()}
            verified = verify_claim_pipeline(claim_obj)
            parsed = {
                "claims": [
                    {
                        "text": verified.get("text", text.strip()),
                        "misp_confidence": float(verified.get("misp_confidence", 0.5)),
                        "short_reason": "Direct Vertex/fact-check/support pipeline result.",
                        "references": verified.get("references", [])
                    }
                ],
                "overall_misp_confidence": float(verified.get("misp_confidence", 0.5)),
                "top_reasons": ["Direct Vertex/fact-check/support pipeline used due to empty model output."]
            }
            explanation_str = json.dumps(parsed, indent=2)
            debug["direct_factcheck"] = True
        except Exception as e:
            logger.debug(f"Direct fact-check fallback failed: {e}")

    # Normalize parsed confidences
    corrected_overall: Optional[float] = None
    if parsed:
        try:
            parsed, corrected_overall = _normalize_parsed_confidences(parsed, explanation_str)
            debug["normalized_overall"] = corrected_overall
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")

    # derive misp_confidence (0..1)
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

                if not existing_refs:
                    query = f'{claim_text} fact check OR debunk OR "fact-check"'
                    hits = _search_web_google_customsearch(query, num=3)
                    existing_refs = hits

                try:
                    c["references"] = [_normalize_ref(r) for r in (existing_refs or [])][:5]
                except Exception:
                    c["references"] = []

                sources_result.append({"claim": claim_text, "sources": c.get("references")})
                for hit in c.get("references", []):
                    link_var = hit.get("link") or hit.get("claim_url") or ""
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
                    hits_norm = [_normalize_ref(h) for h in hits][:5]
                    sources_result.append({"claim": query, "sources": hits_norm})
                    for hit in hits_norm:
                        link_var = hit.get("link")
                        if link_var and link_var not in [t.get("link") for t in top_links]:
                            top_links.append(hit)
    except Exception as e:
        logger.warning(f"Source search failed: {e}")

    # -----------------
    # Build user-facing explanation (priority: fact-check verdicts)
    # -----------------
    user_explanation_parts: List[str] = []

    fc_decisive: Optional[Dict[str, Any]] = None
    if parsed and isinstance(parsed.get("claims"), list):
        for c in parsed["claims"]:
            refs = c.get("references") or []
            fc_candidates = [r for r in refs if (r.get("claim_url") or "").strip() or ("fact-check" in (r.get("title") or "").lower())]
            if fc_candidates:
                try:
                    fc_hits = _factcheck_search(c.get("text", ""), num=3)
                    decisive = _pick_decisive_factcheck(fc_hits)
                    if decisive:
                        fc_decisive = decisive
                        if decisive.get("verdict") == "false":
                            c["misp_confidence"] = 0.0
                        elif decisive.get("verdict") == "true":
                            c["misp_confidence"] = 1.0
                        c["references"] = [_normalize_ref(h) for h in (fc_hits or refs)][:5]
                        break
                except Exception:
                    pass

    if fc_decisive:
        verdict = fc_decisive.get("verdict")
        publisher = fc_decisive.get("publisher") or "a fact-checker"
        claim_url = fc_decisive.get("claim_url") or fc_decisive.get("link") or ""
        textual = fc_decisive.get("textualRating") or fc_decisive.get("snippet") or ""
        if verdict == "false":
            user_explanation_parts.append(f"Fact-check verdict: FALSE — {publisher} ({claim_url}).")
            if textual:
                user_explanation_parts.append(f"Reason: {textual}.")
            user_explanation_parts.append("Recommended action: Do not share; consider reporting to relevant platforms or authorities.")
        elif verdict == "true":
            user_explanation_parts.append(f"Fact-check verdict: TRUE — {publisher} ({claim_url}).")
            if textual:
                user_explanation_parts.append(f"Note: {textual}.")
            user_explanation_parts.append("Recommended action: Content appears to be supported by authoritative sources.")
        else:
            user_explanation_parts.append(f"A fact-check entry was found from {publisher} ({claim_url}) but the verdict is not explicit.")
            user_explanation_parts.append("Recommended action: Verify before sharing; review the linked fact-check for details.")
    else:
        if parsed and parsed.get("claims"):
            claim_count = len(parsed["claims"])
            user_explanation_parts.append(f"I extracted {claim_count} claim(s) from the content and checked them against web sources.")
            for c in parsed["claims"][:2]:
                ct = (c.get("text") or "").replace("\n", " ").strip()
                if len(ct) > 140:
                    ct = ct[:140].strip() + "..."
                user_explanation_parts.append(f"• \"{ct}\" — confidence: {float(c.get('misp_confidence', 0.5)):.2f}")
        else:
            if ocr_text:
                snippet = ocr_text.replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:200].strip() + "..."
                user_explanation_parts.append(f"OCR detected text: \"{snippet}\"")
            else:
                user_explanation_parts.append("No explicit factual claims were reliably extracted from this content.")

        if top_links:
            user_explanation_parts.append(f"I found {len(top_links)} supporting/contradicting sources; review top source(s): {', '.join([t.get('title') or t.get('link') for t in top_links[:2]])}.")
        else:
            user_explanation_parts.append("No authoritative fact-checks were found for these claims.")

        if misp_conf <= 0.3:
            user_explanation_parts.append("Recommended action: Do not share; consider reporting.")
        elif misp_conf <= 0.6:
            user_explanation_parts.append("Recommended action: Verify further before sharing.")
        else:
            user_explanation_parts.append("Recommended action: Content appears safe to share, but verify context if unsure.")

    user_explanation = " ".join(user_explanation_parts[:6])

    # --- finalize result using compute_misinfo_score ---
    try:
        # import here to avoid circular import at module load
        from app.services.analysis_service import compute_misinfo_score
        misinfo = compute_misinfo_score(
            text_signal=float(misp_conf),
            image_safe_search=vision_result.get("safe_search_score"),
            image_labels=image_labels,
            ocr_text=ocr_text,
            llm_debug=debug.get("vertex_raw")
        )
    except Exception as e:
        logger.debug(f"compute_misinfo_score failed: {e}")
        # fallback mapping: trustworthiness = misp_conf
        trust = float(max(0.0, min(1.0, misp_conf)))
        misinfo = {"score": round(trust * 10.0, 2), "color": map_confidence_to_color(trust), "top_reasons": []}

    # normalize top_links
    norm_top_links = [_normalize_ref(t) for t in top_links][:5]

    # normalize per-claim fields (add 0..10 variant)
    if parsed and isinstance(parsed.get("claims"), list):
        for c in parsed["claims"]:
            refs = c.get("references") or []
            c["references"] = [_normalize_ref(r) for r in refs][:5]
            misp_val = float(c.get("misp_confidence", 0.5))
            c["misp_confidence_0_1"] = misp_val
            c["confidence_0_10"] = round(misp_val * 10.0, 2)

    # final payload
    result = {
        "score": float(misinfo.get("score", 5.0)),
        "color": misinfo.get("color", map_confidence_to_color(misp_conf)),
        "top_reasons": misinfo.get("top_reasons", []),
        "user_explanation": user_explanation,
        "top_sources": norm_top_links,
        "parsed": parsed or {"claims": [], "overall_misp_confidence": None},
        "debug": debug,
        "misp_confidence": float(misp_conf)  # top-level numeric 0..1
    }

    return result


# map to color (small helper)
def map_confidence_to_color(misp_confidence: float) -> str:
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
        msg['From'] = smtp_user or "no-reply@example.com"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(str(smtp_host), int(smtp_port)) as server:
            server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"Report email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send report email: {e}")


# -----------------
# Reporting helper: build a human-friendly report and trigger email if severe
# -----------------
def _build_report_body(analysis_result: Dict[str, Any], reporter: str = "misinfo-bot") -> str:
    misp_conf = analysis_result.get("misp_confidence", 0.5)
    color = analysis_result.get("color", map_confidence_to_color(misp_conf))
    user_explanation = analysis_result.get("user_explanation", "") or ""
    parsed = analysis_result.get("parsed") or {}
    top_sources = analysis_result.get("top_sources") or []
    debug = analysis_result.get("debug") or {}

    ts = datetime.utcnow().isoformat() + "Z"
    header = f"Misinformation incident report — {ts}\nReporter: {reporter}\nSeverity (0..1): {misp_conf:.3f}  Color: {color}\n\n"

    # claims detail
    claims_texts = []
    for idx, c in enumerate(parsed.get("claims", []) or []):
        text = (c.get("text") or "").replace("\n", " ").strip()
        conf = float(c.get("misp_confidence", 0.5))
        refs = c.get("references") or []
        ref_lines = []
        for r in refs[:5]:
            title = r.get("title") or r.get("publisher") or ""
            link = r.get("link") or r.get("claim_url") or r.get("url") or ""
            snippet = (r.get("snippet") or "")[:200].replace("\n", " ")
            ref_lines.append(f"- {title} | {link}\n  {snippet}")
        claims_texts.append(f"Claim #{idx+1} (confidence={conf:.2f}):\n{text}\nReferences:\n" + ("\n".join(ref_lines) if ref_lines else "  (none found)") + "\n")

    if not claims_texts:
        claims_texts = ["No structured claims were identified by the pipeline."]

    # top sources de-dup and format
    top_src_lines = []
    seen_links = set()
    for s in top_sources:
        l = s.get("link") or s.get("claim_url") or s.get("url") or ""
        if not l or l in seen_links:
            continue
        seen_links.add(l)
        top_src_lines.append(f"- {s.get('title') or l} | {l}")

    if not top_src_lines:
        top_src_lines = ["(no top sources)"]

    # small debug excerpt
    debug_excerpt = ""
    if debug:
        raw = str(debug.get("vertex_raw") or debug.get("vertex_raw", ""))[:2000]
        debug_excerpt = f"\n\n--- debug excerpt (vertex_raw) ---\n{raw}\n"

    body = header
    body += "Summary / user_explanation:\n" + textwrap.fill(user_explanation, width=100) + "\n\n"
    body += "Claims found (top 5):\n" + "\n".join(claims_texts[:5]) + "\n"
    body += "Top sources discovered:\n" + "\n".join(top_src_lines[:10]) + "\n"
    body += debug_excerpt
    body += "\n\nFull analysis JSON (abridged) attached below:\n"
    try:
        body += json.dumps(analysis_result, indent=2)[:8000]
    except Exception:
        body += "(failed to encode full JSON)\n"
    return body


def report_if_severe(analysis_result: Dict[str, Any], report_to_email: Optional[str] = None, severity_threshold_score: float = 4.0) -> None:
    """
    analysis_result: the dict returned by analyze_content (contains score 0..10)
    severity_threshold_score: 0..10 where lower means more severe. If computed score < threshold, send report.
    """
    try:
        misp_conf = float(analysis_result.get("misp_confidence", 0.5))
    except Exception:
        misp_conf = 0.5

    # convert to trustworthiness 0..10 (where higher = more truthful)
    trustworthiness = float(max(0.0, min(1.0, misp_conf)))
    score_0_10 = round(trustworthiness * 10.0, 2)

    severity_score = score_0_10

    if severity_score < severity_threshold_score:
        to_email = report_to_email or getattr(settings, "REPORT_TO_EMAIL", None) or settings.SMTP_USER or None
        if not to_email:
            logger.warning("report_if_severe: no target REPORT_TO_EMAIL configured; skipping email send.")
        subject = f"[Misinformation Alert] severity {severity_score:.2f}/10 - color {analysis_result.get('color')}"
        body = _build_report_body(analysis_result, reporter="backend-misinfo-detector")
        try:
            incident_log = {
                "event": "misinformation_detected",
                "severity_score": severity_score,
                "color": analysis_result.get("color"),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "top_sources_count": len(analysis_result.get("top_sources", []) or []),
                "claims_count": len((analysis_result.get("parsed") or {}).get("claims", []) or [])
            }
            logger.warning("Misinformation incident: %s", json.dumps(incident_log))
        except Exception:
            logger.warning("Misinformation incident (no JSONable log)")

        if to_email:
            try:
                send_misinformation_report(subject, body, to_email)
            except Exception as e:
                logger.error(f"Failed to send incident report email: {e}")
    else:
        logger.info("No report created: severity_score=%s (threshold=%s)", severity_score, severity_threshold_score)
