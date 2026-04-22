"""
CLI smoke-test entry point.
On Day 1, everyone clones and runs this to confirm their environment is OK.

Usage:
  python main.py          # Interactive mode (uses mock)
  python main.py --auto   # Runs 3 automated turns
"""
import sys

from config import CONFIG
from core.generator import StoryEngine
from retrieval.world_kb import WorldKB
from memory.short_term import ShortTermMemory
from memory.summarizer import Summarizer


def build_engine() -> StoryEngine:
    print(f"=== AI Dungeon smoke test ===")
    print(f"USE_MOCK_LLM = {CONFIG.USE_MOCK_LLM}")
    print(f"USE_RAG      = {CONFIG.USE_RAG}")
    print(f"USE_MEMORY   = {CONFIG.USE_MEMORY}")
    print(f"LLM_MODEL    = {CONFIG.LLM_MODEL}")
    print()

    world_kb = WorldKB()
    memory = ShortTermMemory()
    summarizer = Summarizer()
    return StoryEngine(world_kb, memory, summarizer)


def print_result(result):
    print(f"\n📖 {result.text}\n")
    if result.choices:
        print("Available actions:")
        for i, c in enumerate(result.choices, 1):
            print(f"  [{i}] {c}")
    if result.state_delta:
        print(f"\nState delta: {result.state_delta}")
    print("-" * 60)


def auto_mode():
    engine = build_engine()
    print(engine.world_kb.get_opening())
    print("-" * 60)

    inputs = [
        "I enter the space station through the airlock",
        "Head toward the control room",
        "Check the logs on the console",
    ]
    for inp in inputs:
        print(f"\nPlayer: {inp}")
        result = engine.generate(inp)
        print_result(result)
    print("\n✅ Automated test passed.")


def interactive_mode():
    engine = build_engine()
    print(engine.world_kb.get_opening())
    print("-" * 60)
    print("Type /quit to exit, /story to generate the final novel.\n")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/story":
            from story.compiler import compile_story
            print("\n" + "=" * 60)
            print("📖 Final novel")
            print("=" * 60)
            print(compile_story(engine.get_full_story_log()))
            continue

        result = engine.generate(user_input)
        print_result(result)


if __name__ == "__main__":
    if "--auto" in sys.argv:
        auto_mode()
    else:
        interactive_mode()
