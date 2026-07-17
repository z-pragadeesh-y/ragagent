"""
Hybrid retrieval: combines BM25 (keyword) + vector (semantic) search via
Reciprocal Rank Fusion, then reranks the fused candidates with a cross-encoder.

Plan 2 Step 1 update: now uses the structure-aware chunks (real section
boundaries, domain_tag/section_title metadata) instead of the old blind
fixed-size chunks. An optional domain_tag filter is supported for when the
router is later extended to also classify domain, not just category - not
wired into the graph yet, but the retrieval layer is ready for it.
"""
import re
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from ingestion.loader import load_documents
from ingestion.structured_chunker import chunk_documents_structured
from ingestion.vectorstore import load_structured_vectorstore

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cached globals so we don't rebuild BM25/reload chunks on every call
_bm25_index = None
_bm25_chunks = None
_reranker = None


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation before splitting, so tokens like '(HyDE)' match 'hyde'."""
    return re.findall(r"\b\w+\b", text.lower())


def _get_bm25_index():
    global _bm25_index, _bm25_chunks
    if _bm25_index is None:
        docs = load_documents()
        _bm25_chunks = chunk_documents_structured(docs)
        tokenized = [_tokenize(c.page_content) for c in _bm25_chunks]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_chunks


def _get_reranker():
    """Loads (once) the local cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
    return _reranker


def reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids, k=60):
    """Combines two ranked lists of chunk IDs into one fused score dict."""
    scores = {}
    for rank, chunk_id in enumerate(vector_ranked_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ranked_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return scores


def hybrid_retrieve(query: str, fusion_k: int = 15, final_k: int = 4, domain_tag: str = None):
    """
    Full pipeline: vector search + BM25 search -> RRF fusion -> cross-encoder rerank -> top final_k.
    If domain_tag is provided, both vector search and the BM25 candidate pool are restricted
    to chunks tagged with that domain, eliminating cross-domain noise entirely for queries
    where the domain is already known (e.g. from router classification).
    """
    vectorstore = load_structured_vectorstore()
    bm25_index, bm25_chunks = _get_bm25_index()

    # 1. Vector search (semantic) - get more candidates than we need
    vector_kwargs = {"k": fusion_k}
    if domain_tag:
        vector_kwargs["filter"] = {"domain_tag": domain_tag}
    vector_results = vectorstore.similarity_search(query, **vector_kwargs)
    vector_ids = [doc.page_content for doc in vector_results]

    # 2. BM25 search (keyword) - restrict candidate pool to the domain if given
    tokenized_query = _tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)

    if domain_tag:
        eligible_indices = [i for i, c in enumerate(bm25_chunks) if c.metadata.get("domain_tag") == domain_tag]
    else:
        eligible_indices = range(len(bm25_chunks))

    bm25_ranked_indices = sorted(eligible_indices, key=lambda i: bm25_scores[i], reverse=True)[:fusion_k]
    bm25_ids = [bm25_chunks[i].page_content for i in bm25_ranked_indices]

    # 3. Fuse rankings via RRF
    fused_scores = reciprocal_rank_fusion(vector_ids, bm25_ids)

    content_to_doc = {doc.page_content: doc for doc in vector_results}
    for i in bm25_ranked_indices:
        chunk = bm25_chunks[i]
        content_to_doc.setdefault(chunk.page_content, chunk)

    fused_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_docs = [content_to_doc[content] for content, _ in fused_candidates if content in content_to_doc]

    # 4. Rerank the fused candidates with a cross-encoder for true relevance
    reranker = _get_reranker()
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs) if pairs else []

    reranked = sorted(zip(candidate_docs, rerank_scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in reranked[:final_k]]

    return top_docs


if __name__ == "__main__":
    query = "What does NIST say organizations should do about AI risks?"
    print(f"Query: {query}\n")

    results = hybrid_retrieve(query)
    for i, doc in enumerate(results):
        print(f"[{i+1}] Source: {doc.metadata.get('source')} | domain: {doc.metadata.get('domain_tag')} | section: {doc.metadata.get('section_title')}")
        print(doc.page_content[:200])
        print()
