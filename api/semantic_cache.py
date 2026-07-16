"""
Simple in-memory semantic cache: embeds incoming queries and compares them
via cosine similarity against recently cached queries. A near-duplicate
question skips the full graph entirely and returns the cached answer.
"""
import time
import numpy as np
from ingestion.embedder import get_embedding_model

SIMILARITY_THRESHOLD = 0.95  # cosine similarity; high bar since wrong cache hits are worse than cache misses
CACHE_TTL_SECONDS = 3600     # 1 hour - answers can go stale, don't cache forever
MAX_CACHE_SIZE = 200         # cap memory usage


class SemanticCache:
    def __init__(self):
        self.embedder = get_embedding_model()
        self.entries = []  # list of dicts: {embedding, question, answer, timestamp}

    def _cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _prune_expired(self):
        now = time.time()
        self.entries = [e for e in self.entries if now - e["timestamp"] < CACHE_TTL_SECONDS]

    def get(self, question: str):
        """Returns cached answer if a near-duplicate question exists, else None."""
        self._prune_expired()
        if not self.entries:
            return None

        query_embedding = self.embedder.embed_query(question)
        best_score, best_entry = 0.0, None
        for entry in self.entries:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score, best_entry = score, entry

        if best_score >= SIMILARITY_THRESHOLD:
            return best_entry["answer"]
        return None

    def set(self, question: str, answer: str):
        """Stores a new question/answer pair in the cache."""
        query_embedding = self.embedder.embed_query(question)
        self.entries.append({
            "embedding": query_embedding,
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
        })
        if len(self.entries) > MAX_CACHE_SIZE:
            self.entries = self.entries[-MAX_CACHE_SIZE:]  # drop oldest


_cache_instance = None

def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance