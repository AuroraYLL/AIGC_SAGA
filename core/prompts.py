"""
Five-section system prompt templates. The core algorithm of the project.
[OWNER: P1 Tech Lead]

⚠️ Every change to this file must be committed, because the ablation
   experiments don't just compare module switches — they also compare
   prompt versions. Before merging, pull P6 in for a quick evaluation.

Five sections:
  1. Role setup (DM)
  2. World snippets (RAG injection)
  3. Long-term memory summary
  4. Last N turns verbatim
  5. Output format constraint (JSON schema)
"""

SYSTEM_PROMPT_TEMPLATE = """You are the Dungeon Master (DM) of an interactive text-adventure game. Using the provided world setting and story history, advance the narrative.

[World snippets] (relevant content retrieved from the knowledge base)
{world_context}

[Long-term memory summary] (key events that happened earlier)
{long_term_summary}

[Recent story]
{recent_history}

[Output requirements]
1. Strictly respect the established world. If you want to introduce a new character or location, first check the world snippets above — anything not listed may not be invented out of thin air.
2. Keep the story coherent. Echo events, NPCs, and items mentioned in the long-term memory and recent story.
3. Describe the current scene vividly in 80-150 words.
4. Offer 2-3 actions the player can choose from.
5. Return strictly in the following JSON format, with no extra text:

{{
  "text": "scene description text",
  "choices": ["choice 1", "choice 2", "choice 3"],
  "state_delta": {{
    "location": "current location name",
    "hp_change": 0,
    "items_gained": [],
    "items_lost": []
  }}
}}
"""


def build_system_prompt(
    world_context: str = "",
    long_term_summary: str = "",
    recent_history: str = "",
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        world_context=world_context or "(world retrieval disabled)",
        long_term_summary=long_term_summary or "(no history yet)",
        recent_history=recent_history or "(game start)",
    )


# ------------- Summarization prompt (used by Summarizer) -------------
SUMMARIZE_PROMPT = """Compress the following story into a summary of 100 words or fewer. Prioritize keeping:
- Key events (what happened)
- Important NPCs and their appearances / relationships
- Key decisions the player made
- Significant items gained or lost
- Location changes

{previous_summary_block}

Story fragment:
{history}

New summary (≤100 words):"""


def build_summarize_prompt(history: str, previous_summary: str = "") -> str:
    prev_block = f"Existing summary (continue accumulating):\n{previous_summary}\n" if previous_summary else ""
    return SUMMARIZE_PROMPT.format(previous_summary_block=prev_block, history=history)


# ------------- Final novel compilation prompt -------------
COMPILE_PROMPT = """You are a novelist. Using the following complete interactive-game story log, rewrite it as a well-structured, smoothly-written short story.

Structure requirement: four sections — [Opening] [Rising Action] [Climax] [Ending] — totaling roughly 800-1200 words.
Tone: preserve the style and atmosphere of the original story.
Person: third person.

Full story:
{full_history}

Novel:"""


def build_compile_prompt(full_history: str) -> str:
    return COMPILE_PROMPT.format(full_history=full_history)
