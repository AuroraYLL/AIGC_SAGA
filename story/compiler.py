"""
Final novel compilation: rewrite the full story log into a structured short story.
[Week 2 task, anyone can pick it up]

Simple enough that it just makes another LLM call, so it lives in its own file.
"""
from core.llm import call_llm
from core.prompts import build_compile_prompt


def compile_story(full_history: str) -> str:
    """
    Args:
        full_history: the output of StoryEngine.get_full_story_log()
    Returns:
        A four-part short story (opening - rising action - climax - ending)
    """
    prompt = build_compile_prompt(full_history)
    return call_llm(
        system_prompt="You are an experienced novelist.",
        user_prompt=prompt,
        json_mode=False,
        purpose="compile",
    )
