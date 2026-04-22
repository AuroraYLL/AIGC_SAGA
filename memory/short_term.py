"""
Short-term memory: the most recent N turns of dialogue.
[OWNER: P3 Memory Engineer]

Public interface:
  memory.add_turn(user_input, assistant_output)
  memory.get_recent_as_text() -> str        # used in the prompt
  memory.should_trigger_summary() -> bool   # whether it's time to compress
  memory.pop_oldest_for_summary() -> List[Turn]  # pop old content for the Summarizer
"""
from dataclasses import dataclass
from typing import List

from config import CONFIG


@dataclass
class Turn:
    user_input: str
    assistant_output: str


class ShortTermMemory:
    """Sliding window: keeps the most recent N turns verbatim. When the
    threshold is exceeded, older turns are handed to the Summarizer for
    compression."""

    def __init__(self):
        self.turns: List[Turn] = []

    # ---- Write ----
    def add_turn(self, user_input: str, assistant_output: str):
        self.turns.append(Turn(user_input, assistant_output))

    # ---- Read ----
    def get_recent_as_text(self) -> str:
        """Return only the most recent N turns, for inclusion in the prompt."""
        recent = self.turns[-CONFIG.SHORT_TERM_TURNS:]
        return self._format_turns(recent)

    def all_turns_as_text(self) -> str:
        """Return the entire history (used by the final-novel compiler)."""
        return self._format_turns(self.turns)

    # ---- Compression trigger ----
    def should_trigger_summary(self) -> bool:
        return len(self.turns) >= CONFIG.SUMMARY_TRIGGER_TURNS

    def pop_oldest_for_summary(self) -> List[Turn]:
        """
        Pop the conversations older than "the most recent N turns". These
        will be summarized, after which their verbatim text is discarded.
        Example: with SHORT_TERM_TURNS=5 and SUMMARY_TRIGGER_TURNS=10, once
        the turn count hits 10 the oldest 5 are popped and compressed,
        while the most recent 5 are kept verbatim.
        """
        keep = CONFIG.SHORT_TERM_TURNS
        if len(self.turns) <= keep:
            return []
        old = self.turns[:-keep]
        self.turns = self.turns[-keep:]
        return old

    # ---- Internal ----
    @staticmethod
    def _format_turns(turns: List[Turn]) -> str:
        if not turns:
            return ""
        return "\n".join(
            f"Player: {t.user_input}\nDM: {t.assistant_output}" for t in turns
        )

    def __len__(self):
        return len(self.turns)
