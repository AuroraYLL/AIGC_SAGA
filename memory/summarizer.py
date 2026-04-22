"""
Long-term memory: compress old story via LLM summarization.
[OWNER: P3 Memory Engineer]

Core idea: short-term memory keeps the most recent N turns verbatim; older
content is compressed by the LLM into a ~100-word summary. When more
content arrives, the existing summary + the new content is compressed
again, so the summary doesn't grow without bound.

This is the project's biggest technical innovation (emphasize it in the report).
"""
from typing import List

from core.llm import call_llm
from core.prompts import build_summarize_prompt
from memory.short_term import Turn


class Summarizer:
    """Cumulative summarization: each compression feeds the existing
    summary back in, so continuity is preserved."""

    def __init__(self):
        self._summary: str = ""

    def get_summary(self) -> str:
        return self._summary

    def update(self, old_turns: List[Turn]):
        """Run one LLM compression pass over the popped old turns, and append to the summary."""
        if not old_turns:
            return

        history_text = "\n".join(
            f"Player: {t.user_input}\nDM: {t.assistant_output}" for t in old_turns
        )
        prompt = build_summarize_prompt(
            history=history_text,
            previous_summary=self._summary,
        )

        new_summary = call_llm(
            system_prompt="You are a professional story-summary assistant, skilled at condensing the key information of a narrative.",
            user_prompt=prompt,
            json_mode=False,
            purpose="summarize",
        )
        self._summary = new_summary.strip()

    def reset(self):
        """Clear on ablation-experiment restart / new game."""
        self._summary = ""
