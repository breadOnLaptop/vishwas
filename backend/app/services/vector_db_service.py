"""
Tiny in-memory vector DB stub for prototyping.

- upsert(item): item must be a dict. If it contains a "vector" key (list/tuple of floats),
  the vector will be stored alongside the item.
- query(query_vec, top_k=5): returns top_k items sorted by cosine similarity (descending).
- clear(): clears the DB.

This is intentionally small — replace with FAISS/Milvus/Pinecone for production.
"""

from typing import List, Dict, Any, Optional
import uuid
import threading
import numpy as np

_db: List[Dict[str, Any]] = []
_lock = threading.Lock()

def _normalize(vec: List[float]) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

def upsert(item: Dict[str, Any]) -> str:
    """
    Insert an item into the in-memory DB.
    The function assigns a generated _id and stores the vector if provided.
    Returns the generated id.
    """
    item_copy = dict(item)  # shallow copy
    item_id = str(uuid.uuid4())
    item_copy["_id"] = item_id

    # normalize vector for cosine similarity convenience
    if "vector" in item_copy and item_copy["vector"] is not None:
        try:
            item_copy["_vector_norm"] = _normalize(item_copy["vector"]).tolist()
        except Exception:
            # remove invalid vector
            item_copy.pop("vector", None)
            item_copy.pop("_vector_norm", None)

    with _lock:
        _db.append(item_copy)

    return item_id

def query(query_vec: Optional[List[float]] = None, top_k: int = 5, include_scores: bool = True) -> List[Dict[str, Any]]:
    """
    Query the in-memory DB.
    - If query_vec is provided and items have stored vectors, returns nearest by cosine similarity.
    - If query_vec is None, returns the most recent `top_k` items.

    Each returned item is a shallow copy with optional "_score" if include_scores=True.
    """
    with _lock:
        items = list(_db)

    if not items:
        return []

    if query_vec is None:
        # return last N items
        res = items[-top_k:]
        return [dict(i) for i in res]

    # prepare normalized query vector
    try:
        qnorm = _normalize(query_vec)
    except Exception:
        return []

    # compute similarities for items that have _vector_norm
    scored: List[Dict[str, Any]] = []
    for it in items:
        vec_norm = it.get("_vector_norm")
        if vec_norm:
            try:
                sim = float(np.dot(qnorm, np.array(vec_norm, dtype=float)))
            except Exception:
                sim = 0.0
            scored.append((sim, it))
    # sort descending by similarity
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, it in scored[:top_k]:
        out = dict(it)
        if include_scores:
            out["_score"] = round(float(sim), 4)
        # remove internal vector norm if present
        out.pop("_vector_norm", None)
        results.append(out)

    return results

def clear():
    """Clear the in-memory DB (for testing)."""
    with _lock:
        _db.clear()
