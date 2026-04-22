"""
Ablation experiment harness.
[OWNER: P6 Evaluation & Report]

Runs 4 configurations: baseline / rag_only / memory_only / full.
Each configuration processes the same set of player inputs and produces
a story log, for blind scoring.

Usage:
  python -m evaluation.ablation            # run all 4 configurations
  python -m evaluation.ablation full       # run only one

⚠️ Real runs consume API (if USE_MOCK_LLM=false).
   Recommendation: first confirm the harness itself is OK with the mock,
   then switch to the real API.
"""
import os
import sys
import json
import importlib
from pathlib import Path
from typing import Dict, List


# The 4 ablation configurations
CONFIGS: Dict[str, Dict[str, str]] = {
    "baseline":    {"USE_RAG": "false", "USE_MEMORY": "false"},
    "rag_only":    {"USE_RAG": "true",  "USE_MEMORY": "false"},
    "memory_only": {"USE_RAG": "false", "USE_MEMORY": "true"},
    "full":        {"USE_RAG": "true",  "USE_MEMORY": "true"},
}


# Fixed test script: all configurations run the same inputs, to ensure a fair comparison.
# P6 may adjust these to match the world, but once the experiment starts they cannot be changed.
FIXED_SCRIPT: List[str] = [
    "I enter the space station through the airlock and take a look around",
    "Head down the main corridor toward the control room",
    "Enter the control room and examine Station Chief Mira's terminal",
    "Pull up the full contents of the last log entry",
    "Leave the control room and head to the research lab",
    "Use the ID card I just picked up to unlock the lab",
    "Search the experiment notebook carefully",
    "Hear a dragging sound coming from the ventilation shaft",
    "Shine the flashlight at the vent",
    "Head toward the engine room to investigate",
]


def _reload_modules():
    """After env vars change, force re-import of every module that reads CONFIG."""
    mods_to_reload = [
        "config", "core.llm", "core.prompts", "core.generator",
        "retrieval.embedder", "retrieval.vector_store", "retrieval.world_kb",
        "memory.short_term", "memory.summarizer",
    ]
    for m in mods_to_reload:
        if m in sys.modules:
            importlib.reload(sys.modules[m])


def run_config(config_name: str, script: List[str]) -> List[dict]:
    """Run one configuration; return the per-turn log."""
    # 1. Set env vars
    for k, v in CONFIGS[config_name].items():
        os.environ[k] = v

    # 2. Force reload of modules (because CONFIG is read at import time)
    _reload_modules()

    # 3. Build the engine
    from core.generator import StoryEngine
    from retrieval.world_kb import WorldKB
    from memory.short_term import ShortTermMemory
    from memory.summarizer import Summarizer

    engine = StoryEngine(WorldKB(), ShortTermMemory(), Summarizer())

    # 4. Run the script
    logs = []
    for turn_idx, user_input in enumerate(script, 1):
        result = engine.generate(user_input)
        logs.append({
            "turn": turn_idx,
            "user_input": user_input,
            "text": result.text,
            "choices": result.choices,
            "state_delta": result.state_delta,
        })
        print(f"  [{config_name}] turn {turn_idx}/{len(script)} ✓")

    return logs


def main(target: str = None):
    out_dir = Path("eval_outputs")
    out_dir.mkdir(exist_ok=True)

    configs_to_run = [target] if target else list(CONFIGS.keys())

    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"Unknown configuration: {name}. Available: {list(CONFIGS.keys())}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Running configuration: {name}")
        print(f"  USE_RAG    = {CONFIGS[name]['USE_RAG']}")
        print(f"  USE_MEMORY = {CONFIGS[name]['USE_MEMORY']}")
        print(f"{'=' * 50}")

        logs = run_config(name, FIXED_SCRIPT)

        out_path = out_dir / f"{name}.json"
        out_path.write_text(
            json.dumps({
                "config": CONFIGS[name],
                "script": FIXED_SCRIPT,
                "logs": logs,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✅ Saved: {out_path}")

    print("\nAll done. Hand eval_outputs/*.json to the graders for blind review.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(target)
