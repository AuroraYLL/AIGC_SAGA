"""
LLM wrapper + mock switch.
[OWNER: P1 Tech Lead]

Key design: when USE_MOCK_LLM=true, hardcoded story beats are returned,
so development costs zero. Switch to the real API only for:
  (1) P1 doing prompt tuning
  (2) Integration tests
  (3) Ablation experiments
  (4) Demos
"""
import json
import random
from typing import Optional

from config import CONFIG


# ------------- Mock data (P4 can work on the UI without a real API) -------------
MOCK_SCENES = [
    {
        "text": "The emergency lights flicker erratically along the corridor. From the control room to your left you hear a sustained electrical hum, and the air carries a faint tang of iron — or blood.",
        "choices": ["Sneak toward the control room", "Go back to the airlock for equipment first", "Hail the station's comms channel"],
        "state_delta": {"location": "Main Corridor", "hp_change": 0, "items_gained": [], "items_lost": []},
    },
    {
        "text": "The console screen is still lit. The last log entry halts on Station Chief Mira's name. The timestamp in the lower-right corner reads 72 hours ago.",
        "choices": ["Read the full log", "Pull up the station's surveillance footage", "Try restarting the comms array"],
        "state_delta": {"location": "Control Room", "hp_change": 0, "items_gained": ["Dead ID Card"], "items_lost": []},
    },
    {
        "text": "Your breath echoes inside the helmet. Something is moving deep inside the ventilation shaft — not mechanical, more like a slow, rhythmic dragging sound.",
        "choices": ["Hide behind the maintenance bay", "Turn on your flashlight to look", "Kill all the lights and hold still"],
        "state_delta": {"location": "Engine Room", "hp_change": -5, "items_gained": [], "items_lost": []},
    },
]


def _mock_response() -> str:
    return json.dumps(random.choice(MOCK_SCENES), ensure_ascii=False)


def _mock_summary() -> str:
    return "The player entered the space station through the airlock, discovered Station Chief Mira's final log in the control room (72 hours ago), and heard strange sounds coming from the engine room."


# ------------- Real API call -------------
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=CONFIG.LLM_API_KEY, base_url=CONFIG.LLM_BASE_URL)
    return _client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    purpose: str = "generate",
) -> str:
    """
    Unified LLM call entry point.

    Args:
        system_prompt: system instruction
        user_prompt: user input
        json_mode: require JSON output (used by the generate() main pipeline)
        purpose: "generate" / "summarize" / "compile", used to pick the mock branch

    Returns:
        Model output string (if json_mode=True, guaranteed to be json.loads-able)
    """
    if CONFIG.USE_MOCK_LLM:
        if purpose == "summarize":
            return _mock_summary()
        if purpose == "compile":
            return "[Opening] The investigator steps into Orion-7.\n[Rising action] The control-room log, the strange sounds in the engine room.\n[Climax] A confrontation with an unknown entity.\n[Ending] Withdrawal, truth in hand."
        return _mock_response()

    client = _get_client()
    kwargs = {
        "model": CONFIG.LLM_MODEL,
        "temperature": CONFIG.LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
