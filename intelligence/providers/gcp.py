import os
import json
import logging
import random
import io
import re
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Force reload of ENV
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    from PyPDF2 import PdfReader
    import docx
    DOC_PARSERS_READY = True
except ImportError:
    DOC_PARSERS_READY = False

try:
    from google.cloud import vision
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from google.oauth2 import service_account
    GCP_LIBS_READY = True
except Exception as e:
    GCP_LIBS_READY = False

from utils import prompts

logger = logging.getLogger(__name__)

class GCPProvider:
    def __init__(self, req_id: int):
        self.req_id = req_id
        self.vision_client = None
        self.vertex_model = None
        self.creds = None
        self._setup()

    def _setup(self):
        if not GCP_LIBS_READY: return
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path: creds_path = str(ROOT_DIR / "google-credentials.json")
        elif creds_path.startswith("./") or creds_path.startswith(".\\"): creds_path = str(ROOT_DIR / creds_path[2:])

        if not os.path.exists(creds_path): return

        try:
            self.creds = service_account.Credentials.from_service_account_file(creds_path)
            project = os.getenv("GCP_PROJECT")
            vertexai.init(project=project, location=os.getenv("GCP_REGION", "us-central1"), credentials=self.creds)
            self.vision_client = vision.ImageAnnotatorClient(credentials=self.creds)
            self.vertex_model = GenerativeModel(os.getenv("TEXT_MODEL_ID", "gemini-1.5-flash"))
            
            # DEEP DIAGNOSTIC
            cx = os.getenv("GOOGLE_SEARCH_CX")
            key = os.getenv("GOOGLE_SEARCH_API_KEY")
            logger.info(f"[{self.req_id}] DIAGNOSTIC: Key={key[:5]}... CX={cx[:5]}...")
            
            test_url = f"https://www.googleapis.com/customsearch/v1?q=test&cx={cx}&key={key}"
            logger.info(f"[{self.req_id}] TEST URL (Paste in browser): {test_url}")
            
            test_res = self._query_custom_search("Google")
            if test_res:
                logger.info(f"[{self.req_id}] RAG STATUS: ONLINE (Found {len(test_res)} items)")
            else:
                logger.error(f"[{self.req_id}] RAG STATUS: ERROR (0 results. Your CX ID might be wrong or 'Search entire web' is not propagating)")
                
        except Exception as e:
            logger.error(f"[{self.req_id}] RAG STATUS: CRITICAL ERROR: {e}")

    def analyze(self, text: str, image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        if not self.vertex_model: return self._mock_analyze()

        extracted_text = self._get_content(text, image_bytes, filename)
        
        search_queries = []
        try:
            kw_prompt = f"Identify 2 factual queries to fact-check this: {extracted_text[:500]}. Return JSON list."
            kw_resp = self.vertex_model.generate_content(kw_prompt)
            search_queries = json.loads(self._clean_json(kw_resp.text))
        except:
            search_queries = [extracted_text[:80]]

        all_fact_checks = []
        all_web_links = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            fact_f = [executor.submit(self._query_fact_check, q) for q in search_queries]
            web_f = [executor.submit(self._query_custom_search, q) for q in search_queries]
            for f in fact_f: all_fact_checks.extend(f.result())
            for w in web_f: all_web_links.extend(w.result())

        unique_facts = list({f['claim_text']: f for f in all_fact_checks}.values())
        unique_web = list({w['link']: w for w in all_web_links}.values())
        
        context = f"CONTENT: {extracted_text}\n\nEVIDENCE: {json.dumps(unique_facts)} {json.dumps(unique_web)}"
        
        try:
            vertex_resp = self.vertex_model.generate_content(prompts.get_analysis_prompt(context))
            data = json.loads(self._clean_json(vertex_resp.text))
            data["fact_checks"] = unique_facts[:5]
            data["web_evidence"] = unique_web[:5]
            data["evidence_weight"] = min(10.0, (len(unique_facts) * 4.0) + (len(unique_web) * 1.5))
            return data
        except Exception as e:
            logger.error(f"[{self.req_id}] SYNTHESIS ERROR: {e}")
            raise e

    def _query_fact_check(self, query: str) -> List[Dict]:
        key = os.getenv("GOOGLE_SEARCH_API_KEY")
        if not key or not query: return []
        try:
            service = build("factchecktools", "v1alpha1", developerKey=key, cache_discovery=False)
            res = service.claims().search(query=query).execute()
            return [{"reviewer": c.get("claimReview",[{}])[0].get("publisher",{}).get("name"), "claim_text": c.get("text"), "textual_rating": c.get("claimReview",[{}])[0].get("textualRating"), "url": c.get("claimReview",[{}])[0].get("url")} for c in res.get("claims", [])]
        except: return []

    def _query_custom_search(self, query: str) -> List[Dict]:
        cx = os.getenv("GOOGLE_SEARCH_CX")
        key = os.getenv("GOOGLE_SEARCH_API_KEY")
        if not cx or not key or not query: return []
        try:
            service = build("customsearch", "v1", developerKey=key, cache_discovery=False)
            res = service.cse().list(q=query, cx=cx, num=3).execute()
            return [{"title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet"), "source": i.get("displayLink")} for i in res.get("items", [])]
        except: return []

    def _get_content(self, text: str, image_bytes: bytes, filename: str) -> str:
        fname = (filename or "").lower()
        from PyPDF2 import PdfReader
        import docx
        if fname.endswith(".pdf"):
            try: return "\n".join([p.extract_text() for p in PdfReader(io.BytesIO(image_bytes)).pages if p.extract_text()])
            except: pass
        if fname.endswith(".docx"):
            try: return "\n".join([p.text for p in docx.Document(io.BytesIO(image_bytes)).paragraphs if p.text])
            except: pass
        if image_bytes and self.vision_client:
            try:
                res = self.vision_client.annotate_image({'image': vision.Image(content=image_bytes), 'features': [{"type_": vision.Feature.Type.TEXT_DETECTION}]})
                return res.text_annotations[0].description if res.text_annotations else ""
            except: pass
        return text

    def _clean_json(self, raw: str) -> str:
        s = raw.strip()
        if "```json" in s: s = s.split("```json")[1].split("```")[0]
        elif "```" in s: s = s.split("```")[1].split("```")[0]
        return s.strip()

    def _mock_analyze(self) -> Dict[str, Any]:
        return {"score": 5.0, "color": "orange", "user_explanation": "Offline."}
