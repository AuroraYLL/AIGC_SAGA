"""
Global config + ablation flags.
[OWNER: P1 Tech Lead]

⚠️ Important: the flags here are the control switches for the ablation
experiments. Business code MUST read through CONFIG; hardcoded bypasses
are forbidden (e.g., directly importing and calling RAG). P6's ablation
experiments depend entirely on this.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool_env(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes")


@dataclass
class Config:
    # ---------- LLM ----------
    # Default to mock mode to save money! Only set to false for integration tests / demos.
    USE_MOCK_LLM: bool = _bool_env("USE_MOCK_LLM", "true")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.8"))

    # ---------- Embedding (local, zero API cost) ----------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ---------- Ablation flags ----------
    # P6 uses these two flags to run four configurations:
    # baseline / rag_only / memory_only / full
    USE_RAG: bool = _bool_env("USE_RAG", "true")
    USE_MEMORY: bool = _bool_env("USE_MEMORY", "true")

    # ---------- Memory ----------
    SHORT_TERM_TURNS: int = int(os.getenv("SHORT_TERM_TURNS", "5"))
    SUMMARY_TRIGGER_TURNS: int = int(os.getenv("SUMMARY_TRIGGER_TURNS", "10"))

    # ---------- Retrieval ----------
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    # ---------- World KB ----------
    WORLD_KB_PATH: str = os.getenv("WORLD_KB_PATH", "world/world_kb.json")


CONFIG = Config()


def reload_config():
    """Needed for ablation experiments: re-read env vars after they change."""
    global CONFIG
    CONFIG = Config()
    return CONFIG
