import os
import logging
import random
from typing import Dict, Any

from v1 import intelligence_pb2, intelligence_pb2_grpc
from providers.gcp import GCPProvider

# Setup structured logger
logger = logging.getLogger(__name__)

class IntelligenceService(intelligence_pb2_grpc.IntelligenceServiceServicer):
    def AnalyzeContent(self, request, context):
        req_id = random.randint(1000, 9999)
        logger.info("="*60)
        logger.info(f"RAG PROTOCOL START | ID: {req_id}")
        
        try:
            provider = GCPProvider(req_id)
            raw_data = provider.analyze(request.text, request.image_bytes, request.filename)
            return self._map_to_proto(raw_data, req_id)

        except Exception as e:
            logger.error(f"[{req_id}] CRITICAL: {e}")
            return intelligence_pb2.AnalyzeResponse(score=0.0, color="red")
        finally:
            logger.info("="*60)

    def _map_to_proto(self, data: Dict[str, Any], req_id: int) -> intelligence_pb2.AnalyzeResponse:
        score = float(data.get("score", 5.0))
        
        resp = intelligence_pb2.AnalyzeResponse(
            score=score,
            color=data.get("color", "orange"),
            top_reasons=data.get("top_reasons", []),
            user_explanation=data.get("user_explanation", ""),
            explanation=data.get("explanation", ""),
            evidence_weight=float(data.get("evidence_weight", 0.0)),
            parsed=intelligence_pb2.ParsedOut(
                overall_misp_confidence=1.0 - (score / 10.0),
                claims=[]
            )
        )
        
        # Map Fact Checks
        for f in data.get("fact_checks", []):
            resp.fact_checks.append(intelligence_pb2.FactCheckMatch(
                claimant=f.get("claimant"),
                claim_text=f.get("claim_text"),
                reviewer=f.get("reviewer"),
                textual_rating=f.get("textual_rating"),
                url=f.get("url")
            ))

        # Map Web Evidence
        for w in data.get("web_evidence", []):
            resp.web_evidence.append(intelligence_pb2.WebEvidence(
                title=w.get("title"),
                link=w.get("link"),
                snippet=w.get("snippet"),
                source=w.get("source")
            ))

        # Map Claims
        if "claims" in data:
            for c in data["claims"]:
                resp.parsed.claims.append(intelligence_pb2.Claim(
                    text=c.get("text"),
                    misp_confidence=float(c.get("confidence", 0.5)),
                    confidence_score=float(c.get("confidence", 0.5)) * 10.0,
                    short_reason=c.get("reason")
                ))
                
        return resp
