"""
FAISS vector store wrapper.
[OWNER: P2 Retrieval Engineer]

Public interface:
  store = VectorStore()
  store.build(texts, metas)
  store.search(query, k=3) -> List[(text, meta, score)]
"""
from typing import List, Tuple, Dict, Any

from retrieval.embedder import embed


class VectorStore:
    """A simple vector store built on FAISS IndexFlatIP. A few hundred
    world-knowledge entries is well within what Flat can handle — no need
    for the complexity of IVF/HNSW."""

    def __init__(self):
        self.index = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []

    def build(self, texts: List[str], metas: List[Dict[str, Any]]):
        """Build the index from scratch."""
        assert len(texts) == len(metas), "texts and metas must be the same length"
        if not texts:
            self.index = None
            self.texts = []
            self.metas = []
            return

        import faiss

        vecs = embed(texts)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # normalized vectors + inner product = cosine similarity
        self.index.add(vecs)
        self.texts = list(texts)
        self.metas = list(metas)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """Return top-k hits: [(text, meta, score), ...]"""
        if self.index is None or not self.texts:
            return []

        q = embed([query])
        k_actual = min(k, len(self.texts))
        scores, idx = self.index.search(q, k_actual)

        out = []
        for rank in range(k_actual):
            i = int(idx[0][rank])
            if i < 0:
                continue
            out.append((self.texts[i], self.metas[i], float(scores[0][rank])))
        return out

    def __len__(self):
        return len(self.texts)
