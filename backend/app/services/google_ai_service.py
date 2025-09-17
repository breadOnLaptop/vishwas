import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from typing import Dict, Any

from google.cloud import vision
from google.cloud.vision_v1 import types

try:
    from vertexai.language_models import TextGenerationModel
    from vertexai import initializer as vertex_initializer
    VERTEX_AVAILABLE = True
except Exception:
    VERTEX_AVAILABLE = False

from app.core.config import settings

VISION_CLIENT = vision.ImageAnnotatorClient.from_service_account_file(settings.GOOGLE_APPLICATION_CREDENTIALS)

def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
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
    if response.text_annotations:
        text = response.text_annotations[0].description

    labels = [ann.description for ann in (response.label_annotations or [])]
    safe_search = {}
    if response.safe_search_annotation:
        s = response.safe_search_annotation
        safe_search = {
            "adult": vision.Likelihood(s.adult).name if hasattr(s, "adult") else str(s.adult),
            "violence": vision.Likelihood(s.violence).name if hasattr(s, "violence") else str(s.violence),
            "racy": vision.Likelihood(s.racy).name if hasattr(s, "racy") else str(s.racy),
        }

    likelihood_map = {
        "UNKNOWN": 0.5,
        "VERY_UNLIKELY": 0.0,
        "UNLIKELY": 0.25,
        "POSSIBLE": 0.5,
        "LIKELY": 0.75,
        "VERY_LIKELY": 1.0
    }
    safe_search_score = 0.0
    if safe_search:
        vals = []
        for v in safe_search.values():
            vals.append(likelihood_map.get(v, 0.5))
        safe_search_score = sum(vals) / max(1, len(vals))

    return {
        "text": text,
        "labels": labels,
        "safe_search": safe_search,
        "safe_search_score": safe_search_score,
        "raw": {}  # keep it empty for now
    }

def analyze_text_with_model(prompt: str) -> Dict[str, Any]:
    if VERTEX_AVAILABLE:
        from vertexai import initializer as vertex_initializer
        vertex_initializer.initialize(project=settings.GCP_PROJECT, location=settings.GCP_REGION)
        model = TextGenerationModel.from_pretrained(settings.TEXT_MODEL_ID)
        response = model.predict(prompt, max_output_tokens=512)
        text_out = str(response) if response else ""
        misp_confidence = 0.5
        for token in text_out.split():
            try:
                if token.startswith("0.") or token.startswith("1.") or token in ("0","1"):
                    misp_confidence = float(token)
                    break
            except:
                continue

        return {"explanation": text_out, "misp_confidence": misp_confidence}
    else:
        text_lower = prompt.lower()
        score = 0.0
        suspicious_keywords = ["miracle", "cure", "shocking", "fake", "unverified", "rumor", "scam"]
        for kw in suspicious_keywords:
            if kw in text_lower:
                score += 0.15
        score = min(score, 0.95)
        explanation = "FALLBACK: heuristic used for local testing. No Vertex AI available."
        return {"explanation": explanation, "misp_confidence": score}

def send_misinformation_report(subject: str, body: str, to_email: str):
    msg = MIMEMultipart()
    msg['From'] = settings.SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASS)
            server.send_message(msg)
        print(f"[INFO] Report email sent to {to_email}")
    except Exception as e:
        print(f"[ERROR] Failed to send report email: {e}")
