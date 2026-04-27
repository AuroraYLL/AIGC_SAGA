"""
LLM wrapper + mock switch.
[OWNER: Backend 1 — Core Architecture & API Engineer]

Key design decisions
--------------------
* USE_MOCK_LLM=true  -> zero API cost, deterministic output for dev/test
* USE_MOCK_LLM=false -> real API call with exponential-back-off retry and
                        robust JSON extraction so malformed model output
                        never crashes the pipeline

Public surface (team contract — DO NOT change signatures without team notice)
-----------------------------------------------------------------------------
    call_llm(system_prompt, user_prompt, json_mode, purpose) -> str
    safe_parse_json(text) -> dict   <- utility consumed by generator.py
"""

import json
import logging
import random
import re
import time
from typing import Any, Dict, Optional

from config import CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock data — rich enough for the frontend and ablation team to work with
# ---------------------------------------------------------------------------

_MOCK_SCENES = [
    {
        "text": (
            "Emergency lights flicker erratically along the corridor. "
            "From the control room to your left comes a sustained electrical hum; "
            "the air carries a faint tang of iron — or blood."
        ),
        "choices": [
            "Sneak toward the control room",
            "Go back to the airlock for equipment first",
            "Hail the station's comms channel",
        ],
        "state_delta": {
            "location": "Main Corridor",
            "hp_change": 0,
            "items_gained": [],
            "items_lost": [],
        },
    },
    {
        "text": (
            "The console screen is still lit. "
            "The last log entry halts on Station Chief Mira's name. "
            "The timestamp in the lower-right corner reads 72 hours ago."
        ),
        "choices": [
            "Read the full log",
            "Pull up the station's surveillance footage",
            "Try restarting the comms array",
        ],
        "state_delta": {
            "location": "Control Room",
            "hp_change": 0,
            "items_gained": ["Dead ID Card"],
            "items_lost": [],
        },
    },
    {
        "text": (
            "Your breath echoes inside the helmet. "
            "Something is moving deep inside the ventilation shaft — "
            "not mechanical, more like a slow, rhythmic dragging sound."
        ),
        "choices": [
            "Hide behind the maintenance bay",
            "Turn on your flashlight to look",
            "Kill all the lights and hold still",
        ],
        "state_delta": {
            "location": "Engine Room",
            "hp_change": -5,
            "items_gained": [],
            "items_lost": [],
        },
    },
    {
        "text": (
            "The med-bay door slides open with a hiss. Overturned trays, "
            "scattered medicine vials. One examination table still has restraints "
            "buckled tight — but no patient. A bloody handprint smears the far wall."
        ),
        "choices": [
            "Search for medical supplies",
            "Follow the blood trail to the storage room",
            "Lock the med-bay door and barricade it",
        ],
        "state_delta": {
            "location": "Med Bay",
            "hp_change": 10,
            "items_gained": ["Stimulant Injector"],
            "items_lost": [],
        },
    },
    {
        "text": (
            "Through the observation dome, a debris field drifts past in slow motion. "
            "Among the wreckage you spot another station's escape pod — its hatch "
            "scorched from the outside. Someone got out. Or something got in."
        ),
        "choices": [
            "Attempt to dock with the escape pod",
            "Scan the debris for radio signals",
            "Seal the observation dome and retreat inward",
        ],
        "state_delta": {
            "location": "Observation Dome",
            "hp_change": 0,
            "items_gained": [],
            "items_lost": [],
        },
    },
]

_MOCK_SUMMARY = (
    "The player boarded Orion-7 through the airlock and found the station "
    "in crisis. In the control room, Station Chief Mira's final log — "
    "timestamped 72 hours ago — hinted at a catastrophic biological event. "
    "Strange rhythmic sounds from the engine room forced a hasty retreat "
    "toward the med bay, where a stimulant injector was recovered."
)

_MOCK_NOVEL = (
    "[Opening]\n"
    "The investigator drifts into Orion-7 through an airlock that groans under "
    "the cold of deep space. Emergency lights paint the corridor blood-orange.\n\n"
    "[Rising Action]\n"
    "A 72-hour-old log left by Station Chief Mira details the first signs of "
    "infection. The engine room answers with something worse than silence — "
    "a slow, rhythmic dragging from inside the vents.\n\n"
    "[Climax]\n"
    "Face to face with the unknown entity in the med bay, the investigator "
    "chooses fight over flight, armed only with a stimulant injector and "
    "a dead woman's ID card.\n\n"
    "[Ending]\n"
    "The escape pod clears the debris field. Truth in hand, Orion-7 shrinks "
    "to a pale dot in the rear camera — and then nothing."
)


def _mock_response(purpose: str) -> str:
    """Return a purpose-appropriate mock string."""
    if purpose == "summarize":
        return _MOCK_SUMMARY
    if purpose == "compile":
        return _MOCK_NOVEL
    return json.dumps(random.choice(_MOCK_SCENES), ensure_ascii=False)


# ---------------------------------------------------------------------------
# JSON robustness utilities
# ---------------------------------------------------------------------------

# Regex patterns tried in order when json.loads() fails outright.
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.DOTALL,
)
_BARE_JSON_RE = re.compile(
    r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    re.DOTALL,
)


def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from a raw LLM response, tolerating common failure modes.

    Attempts (in order):
      1. Direct json.loads() — handles well-formed output.
      2. Regex extraction of a markdown code fence (```json ... ```) —
         handles models that wrap their JSON in a code block.
      3. Regex extraction of the first bare {...} block —
         handles models that prepend/append explanatory prose.
      4. Returns an empty dict and logs a warning on total failure.

    Team usage:
        # In generator.py instead of bare json.loads():
        data = safe_parse_json(raw_llm_output)
    """
    if not text:
        return {}

    # Attempt 1 — vanilla parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2 — code-fence extraction
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3 — bare brace extraction (outermost {...})
    bare_match = _BARE_JSON_RE.search(text)
    if bare_match:
        try:
            return json.loads(bare_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(
        "[llm] safe_parse_json: all extraction attempts failed. "
        "Returning empty dict. Raw (first 300 chars): %s",
        text[:300],
    )
    return {}


# ---------------------------------------------------------------------------
# Real API client (lazy-initialized singleton)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # lazy import keeps mock-mode startup fast

        if not CONFIG.LLM_API_KEY:
            raise ValueError(
                "[llm] LLM_API_KEY is not set. "
                "Copy .env.example → .env and fill in your key, "
                "or set USE_MOCK_LLM=true to avoid the real API."
            )
        _client = OpenAI(api_key=CONFIG.LLM_API_KEY, base_url=CONFIG.LLM_BASE_URL)
        logger.info(
            "[llm] OpenAI client initialized. model=%s  base_url=%s",
            CONFIG.LLM_MODEL,
            CONFIG.LLM_BASE_URL,
        )
    return _client


def _call_real_api(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool,
    max_retries: int = 3,
) -> str:
    """
    Call the real LLM endpoint with exponential back-off retry.

    Retries on:
        - Rate-limit errors  (HTTP 429) — waits 2^attempt seconds
        - Transient server errors (HTTP 5xx)
        - Network timeouts / connection drops

    Does NOT retry on:
        - Auth errors (401 / 403) — surface the problem immediately
        - Bad-request errors (400) — usually a prompt bug, not transient
    """
    import openai

    client = _get_client()
    kwargs: Dict[str, Any] = {
        "model": CONFIG.LLM_MODEL,
        "temperature": CONFIG.LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            logger.debug("[llm] API success on attempt %d.", attempt)
            return content

        except openai.RateLimitError as exc:
            wait = 2 ** attempt  # 2 s, 4 s, 8 s
            logger.warning(
                "[llm] Rate-limit (attempt %d/%d). Retrying in %ds. %s",
                attempt, max_retries, wait, exc,
            )
            last_exc = exc
            time.sleep(wait)

        except openai.APIStatusError as exc:
            if exc.status_code in (401, 403, 400):
                logger.error("[llm] Non-retryable API error (%d): %s", exc.status_code, exc)
                raise
            wait = 2 ** attempt
            logger.warning(
                "[llm] Server error %d (attempt %d/%d). Retrying in %ds.",
                exc.status_code, attempt, max_retries, wait,
            )
            last_exc = exc
            time.sleep(wait)

        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            wait = 2 ** attempt
            logger.warning(
                "[llm] Connection/timeout (attempt %d/%d). Retrying in %ds. %s",
                attempt, max_retries, wait, exc,
            )
            last_exc = exc
            time.sleep(wait)

    raise RuntimeError(
        f"[llm] All {max_retries} API attempts failed."
    ) from last_exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    purpose: str = "generate",
) -> str:
    """
    Unified LLM call entry point.  Team contract — do not change the signature.

    Args:
        system_prompt : System instruction fed to the model.
        user_prompt   : Player / user input.
        json_mode     : When True, the caller expects a JSON-parseable string.
                        Mock mode always satisfies this.
                        Real-API mode sets response_format=json_object AND
                        post-processes the output through safe_parse_json(),
                        guaranteeing the returned string is json.loads()-able.
        purpose       : "generate" | "summarize" | "compile"
                        Selects the correct mock branch and is used for logging.

    Returns:
        A plain string.  When json_mode=True, that string is always valid JSON
        (falls back to '{"text": <raw>, "choices": [], "state_delta": {}}' on
        irrecoverable parse failure — the pipeline never raises on bad output).
    """
    logger.debug(
        "[llm] call_llm  purpose=%-10s  json_mode=%s  mock=%s",
        purpose, json_mode, CONFIG.USE_MOCK_LLM,
    )

    # ── Mock path ────────────────────────────────────────────────────────────
    if CONFIG.USE_MOCK_LLM:
        return _mock_response(purpose)

    # ── Real API path ────────────────────────────────────────────────────────
    raw = _call_real_api(system_prompt, user_prompt, json_mode)

    if json_mode:
        # Even with response_format=json_object some providers occasionally
        # wrap their output in markdown fences or add explanatory prose.
        # safe_parse_json() handles all known failure modes.
        parsed = safe_parse_json(raw)
        if not parsed:
            logger.error(
                "[llm] json_mode=True but safe_parse_json returned empty. "
                "Wrapping raw text as fallback JSON. Raw (first 500 chars): %s",
                raw[:500],
            )
            return json.dumps(
                {"text": raw, "choices": [], "state_delta": {}},
                ensure_ascii=False,
            )
        return json.dumps(parsed, ensure_ascii=False)

    return raw
