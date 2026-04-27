"""
Global configuration & ablation flags.
[OWNER: Backend 1 - Core Architecture & API Engineer]

Design rules
------------
* All environment-variable reading happens here and ONLY here.
  Business code must read from CONFIG; no module may call os.getenv() directly.
* The ablation flags (USE_RAG, USE_MEMORY) are the sole control plane for P6's
  four experimental conditions (baseline / rag_only / memory_only / full).
  Any hardcoded bypass of these flags breaks the experiment.
* reload_config() re-reads the environment; used by ablation tests that mutate
  os.environ between runs.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # load .env (if present) before reading env vars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_env(key: str, default: str = "true") -> bool:
    """Parse a boolean environment variable (accepts true/1/yes, case-insensitive)."""
    return os.getenv(key, default).strip().lower() in ("true", "1", "yes")


def _float_env(key: str, default: str) -> float:
    try:
        return float(os.getenv(key, default))
    except ValueError:
        return float(default)


def _int_env(key: str, default: str) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return int(default)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── LLM ─────────────────────────────────────────────────────────────────
    # Default: mock mode ON → zero API cost during development.
    # Switch USE_MOCK_LLM=false only for:
    #   (a) integration tests   (b) demos   (c) ablation experiments
    USE_MOCK_LLM: bool  = field(default_factory=lambda: _bool_env("USE_MOCK_LLM", "true"))

    LLM_MODEL      : str   = field(default_factory=lambda: os.getenv("LLM_MODEL",       "deepseek-chat"))
    LLM_API_KEY    : str   = field(default_factory=lambda: os.getenv("LLM_API_KEY",      ""))
    LLM_BASE_URL   : str   = field(default_factory=lambda: os.getenv("LLM_BASE_URL",     "https://api.deepseek.com"))
    LLM_TEMPERATURE: float = field(default_factory=lambda: _float_env("LLM_TEMPERATURE", "0.8"))

    # ── Embedding (local model, zero API cost) ───────────────────────────────
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # ── Ablation flags ───────────────────────────────────────────────────────
    # P6 toggles these to produce the four experimental conditions:
    #   USE_RAG=false  USE_MEMORY=false → baseline
    #   USE_RAG=true   USE_MEMORY=false → rag_only
    #   USE_RAG=false  USE_MEMORY=true  → memory_only
    #   USE_RAG=true   USE_MEMORY=true  → full  (default)
    USE_RAG   : bool = field(default_factory=lambda: _bool_env("USE_RAG",    "true"))
    USE_MEMORY: bool = field(default_factory=lambda: _bool_env("USE_MEMORY", "true"))

    # ── Memory ───────────────────────────────────────────────────────────────
    # Number of turns kept verbatim in the prompt.
    SHORT_TERM_TURNS   : int = field(default_factory=lambda: _int_env("SHORT_TERM_TURNS",    "5"))
    # Once total turns reach this threshold, the oldest turns are summarized.
    SUMMARY_TRIGGER_TURNS: int = field(default_factory=lambda: _int_env("SUMMARY_TRIGGER_TURNS", "10"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = field(default_factory=lambda: _int_env("RETRIEVAL_TOP_K", "3"))

    # ── World KB ─────────────────────────────────────────────────────────────
    WORLD_KB_PATH: str = field(default_factory=lambda: os.getenv("WORLD_KB_PATH", "world/world_kb.json"))

    # ── Sanity checks ────────────────────────────────────────────────────────
    def __post_init__(self):
        if self.LLM_TEMPERATURE < 0 or self.LLM_TEMPERATURE > 2:
            raise ValueError(
                f"LLM_TEMPERATURE must be in [0, 2], got {self.LLM_TEMPERATURE}"
            )
        if self.SHORT_TERM_TURNS < 1:
            raise ValueError("SHORT_TERM_TURNS must be ≥ 1")
        if self.SUMMARY_TRIGGER_TURNS <= self.SHORT_TERM_TURNS:
            raise ValueError(
                "SUMMARY_TRIGGER_TURNS must be > SHORT_TERM_TURNS "
                f"(got {self.SUMMARY_TRIGGER_TURNS} ≤ {self.SHORT_TERM_TURNS})"
            )

    def describe(self) -> str:
        """
        Return a human-readable summary of the active configuration.
        Useful for logging at startup and for ablation-experiment records.

        Example:
            print(CONFIG.describe())
        """
        api_status = "MOCK" if self.USE_MOCK_LLM else f"REAL ({self.LLM_MODEL} @ {self.LLM_BASE_URL})"
        key_hint = (
            "(not set)" if not self.LLM_API_KEY
            else f"***{self.LLM_API_KEY[-4:]}"  # show only last 4 chars
        )
        rag_status    = "ON" if self.USE_RAG    else "OFF"
        memory_status = "ON" if self.USE_MEMORY else "OFF"
        condition = {
            (True,  True ): "full",
            (True,  False): "rag_only",
            (False, True ): "memory_only",
            (False, False): "baseline",
        }[(self.USE_RAG, self.USE_MEMORY)]

        return (
            f"╔══ CONFIG ══════════════════════════════════╗\n"
            f"║  LLM            : {api_status}\n"
            f"║  API key hint   : {key_hint}\n"
            f"║  Temperature    : {self.LLM_TEMPERATURE}\n"
            f"║  Embedding      : {self.EMBEDDING_MODEL}\n"
            f"╠══ Ablation ════════════════════════════════╣\n"
            f"║  Condition      : {condition}\n"
            f"║  USE_RAG        : {rag_status}\n"
            f"║  USE_MEMORY     : {memory_status}\n"
            f"╠══ Hyperparams ═════════════════════════════╣\n"
            f"║  Short-term     : {self.SHORT_TERM_TURNS} turns verbatim\n"
            f"║  Summary trigger: every {self.SUMMARY_TRIGGER_TURNS} turns\n"
            f"║  Retrieval top-k: {self.RETRIEVAL_TOP_K}\n"
            f"║  World KB path  : {self.WORLD_KB_PATH}\n"
            f"╚════════════════════════════════════════════╝"
        )


# ---------------------------------------------------------------------------
# Singleton & hot-reload
# ---------------------------------------------------------------------------

CONFIG = Config()


def reload_config() -> Config:
    """
    Re-read all environment variables and replace the CONFIG singleton.

    Required by ablation experiments (evaluation/ablation.py) which mutate
    os.environ between the four experimental conditions and then call this
    function to pick up the changes.

    Returns the new Config object (same as CONFIG after the call).
    """
    global CONFIG
    CONFIG = Config()
    return CONFIG
