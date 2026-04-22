"""
World Knowledge Base: load JSON → flatten to embeddable text → build index → retrieve.
[OWNER: P2 Retrieval Engineer]

Public interface:
  kb = WorldKB()
  kb.retrieve_as_text(query, k=3) -> str  # prompt-ready text
"""
import json
from pathlib import Path
from typing import List, Dict, Any

from config import CONFIG
from retrieval.vector_store import VectorStore


# The five top-level categories that P5 must follow when authoring the world
CATEGORIES = ["locations", "characters", "factions", "items", "rules"]


class WorldKB:
    def __init__(self, json_path: str = None):
        path = json_path or CONFIG.WORLD_KB_PATH
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = self.data.get("meta", {})
        self.store = VectorStore()
        self._build_index()

    def _build_index(self):
        """Flatten entries from all 5 categories into single-line texts and embed them into FAISS."""
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for category in CATEGORIES:
            for entry in self.data.get(category, []):
                name = entry.get("name", "Unnamed")
                desc = entry.get("description", "")
                # Splicing tags into the retrievable text too, to help recall
                tags = entry.get("tags", [])
                tag_str = f" (tags: {', '.join(tags)})" if tags else ""
                text = f"[{category}] {name}{tag_str}: {desc}"

                texts.append(text)
                metas.append({
                    "category": category,
                    "id": entry.get("id", name),
                    "name": name,
                })

        self.store.build(texts, metas)
        print(f"[WorldKB] Indexed {len(texts)} world-knowledge entries.")

    def retrieve(self, query: str, k: int = 3):
        return self.store.search(query, k=k)

    def retrieve_as_text(self, query: str, k: int = 3) -> str:
        """Formatted output for use in a prompt."""
        hits = self.store.search(query, k=k)
        if not hits:
            return ""
        lines = [f"- {text}" for text, _meta, _score in hits]
        return "\n".join(lines)

    def get_opening(self) -> str:
        """Return the world's opening paragraph; shown at the start of the game."""
        return self.meta.get("opening", "Game start.")
