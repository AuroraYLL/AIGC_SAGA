"""
Local embeddings. Zero API cost.
[OWNER: P2 Retrieval Engineer]

Defaults to bge-small-en-v1.5 (English-optimized, ~100MB, CPU-friendly).
First launch downloads the model; after that, it's served from local cache.
"""
from typing import List
import numpy as np

from config import CONFIG


_model = None


def get_embedder():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading model {CONFIG.EMBEDDING_MODEL}...")
        _model = SentenceTransformer(CONFIG.EMBEDDING_MODEL)
        print(f"[embedder] Model loaded.")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """
    Text list → normalized embedding matrix (N, D).
    After normalization, dot product equals cosine similarity, so FAISS
    IndexFlatIP works directly.
    """
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)
    vecs = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)
