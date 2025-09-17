"""
Tiny in-memory vector DB stub.
Later this will be replaced with a proper vector DB (FAISS, Milvus, Pinecone, or local FAISS).
"""
import uuid
_db = []

def upsert(item: dict):
    item_id = str(uuid.uuid4())
    item["_id"] = item_id
    _db.append(item)
    return item_id

def query(query_vec, top_k=5):
    # placeholder: return last N items
    return _db[-top_k:]
