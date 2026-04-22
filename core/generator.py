"""
Main pipeline StoryEngine.
[OWNER: P1 Tech Lead]

🔒 The function signatures here are a team-wide contract. DO NOT modify
   them unilaterally! Any change must first be announced to P2/P3/P4/P6
   in the group chat.

Pipeline:
  user input → [RAG retrieval] → [Memory concat] → Prompt → LLM → parse → [update Memory] → return
  Steps in [] can be disabled via ablation flags.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config import CONFIG
from core.llm import call_llm
from core.prompts import build_system_prompt


# ========== Contract types (DO NOT change!) ==========
@dataclass
class GenerateResult:
    """Main-pipeline return type. P4's frontend consumes this directly."""
    text: str
    choices: List[str] = field(default_factory=list)
    state_delta: Dict[str, Any] = field(default_factory=dict)


class StoryEngine:
    """Main pipeline engine. Three modules are injected at construction
    time (dependency inversion, which makes unit tests / ablations easy)."""

    def __init__(self, world_kb, memory, summarizer):
        """
        Args:
            world_kb: retrieval.world_kb.WorldKB instance (provided by P2)
            memory: memory.short_term.ShortTermMemory instance (provided by P3)
            summarizer: memory.summarizer.Summarizer instance (provided by P3)
        """
        self.world_kb = world_kb
        self.memory = memory
        self.summarizer = summarizer

    # ========== Core contract method ==========
    def generate(self, user_input: str) -> GenerateResult:
        """
        Main pipeline. One user input → one story output.

        Args:
            user_input: player's text input (or the text of a chosen option)

        Returns:
            GenerateResult(text, choices, state_delta)
        """
        # --- 1. World retrieval (ablation-disableable) ---
        world_context = ""
        if CONFIG.USE_RAG and self.world_kb is not None:
            world_context = self.world_kb.retrieve_as_text(
                user_input, k=CONFIG.RETRIEVAL_TOP_K
            )

        # --- 2. Memory (ablation-disableable) ---
        long_term_summary = ""
        recent_history = ""
        if CONFIG.USE_MEMORY and self.memory is not None:
            long_term_summary = self.summarizer.get_summary() if self.summarizer else ""
            recent_history = self.memory.get_recent_as_text()

        # --- 3. Build prompt ---
        system_prompt = build_system_prompt(
            world_context=world_context,
            long_term_summary=long_term_summary,
            recent_history=recent_history,
        )

        # --- 4. Call LLM ---
        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_input,
            json_mode=True,
            purpose="generate",
        )

        # --- 5. Parse (robust to LLM occasionally ignoring the format) ---
        result = self._parse_result(raw)

        # --- 6. Update memory (ablation-disableable) ---
        if CONFIG.USE_MEMORY and self.memory is not None:
            self.memory.add_turn(user_input, result.text)
            if self.memory.should_trigger_summary() and self.summarizer is not None:
                old_turns = self.memory.pop_oldest_for_summary()
                self.summarizer.update(old_turns)

        return result

    @staticmethod
    def _parse_result(raw: str) -> GenerateResult:
        """Parse the LLM's JSON-string response into a GenerateResult, with fallback."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat the raw text as the description, no choices.
            return GenerateResult(text=raw or "(System error: no response)", choices=[], state_delta={})

        return GenerateResult(
            text=str(data.get("text", "")),
            choices=list(data.get("choices", [])),
            state_delta=dict(data.get("state_delta", {})),
        )

    # ========== Helper methods ==========
    def get_full_story_log(self) -> str:
        """Used by story/compiler.py: obtain full history (summary + recent)."""
        summary = ""
        recent = ""
        if self.summarizer is not None:
            summary = self.summarizer.get_summary()
        if self.memory is not None:
            recent = self.memory.all_turns_as_text()
        return f"[Long-term memory summary]\n{summary}\n\n[Full dialogue log]\n{recent}"
