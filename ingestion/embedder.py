"""
Loads a local (free) embedding model and generates embeddings for chunks.
"""
import torch
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model():
    """Returns a local HuggingFace embedding model, using GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print("Downloading/loading embedding model (first run downloads ~80MB)...")
    embedder = get_embedding_model()

    # Test on a handful of chunks first, not all 1193 (fast sanity check)
    sample_texts = [c.page_content for c in chunks[:3]]
    vectors = embedder.embed_documents(sample_texts)

    print(f"\nGenerated {len(vectors)} sample embeddings")
    print(f"Embedding dimension: {len(vectors[0])}")
    print(f"First 5 values of vector #0: {vectors[0][:5]}")