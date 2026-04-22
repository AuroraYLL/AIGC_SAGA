"""
Smoke tests. Run before every commit:
  pytest tests/

Ensures the main pipeline doesn't explode in mock mode.
"""
import os

# Make sure tests use the mock, so no money is spent
os.environ["USE_MOCK_LLM"] = "true"
os.environ["USE_RAG"] = "true"
os.environ["USE_MEMORY"] = "true"


def _build():
    from config import reload_config
    reload_config()
    from core.generator import StoryEngine
    from retrieval.world_kb import WorldKB
    from memory.short_term import ShortTermMemory
    from memory.summarizer import Summarizer
    return StoryEngine(WorldKB(), ShortTermMemory(), Summarizer())


def test_contract_generate_returns_result():
    """Main-pipeline contract: generate() must return an object with text / choices / state_delta."""
    engine = _build()
    result = engine.generate("I walk toward the control room")
    assert hasattr(result, "text")
    assert hasattr(result, "choices")
    assert hasattr(result, "state_delta")
    assert isinstance(result.text, str)
    assert isinstance(result.choices, list)
    assert isinstance(result.state_delta, dict)


def test_multiple_turns_dont_crash():
    """Consecutive multi-turn inputs must not crash."""
    engine = _build()
    for i in range(12):  # crosses SUMMARY_TRIGGER_TURNS
        result = engine.generate(f"Turn {i} input")
        assert result.text


def test_ablation_flags_work():
    """RAG / Memory can be independently disabled."""
    os.environ["USE_RAG"] = "false"
    os.environ["USE_MEMORY"] = "false"
    from config import reload_config
    reload_config()

    engine = _build()
    result = engine.generate("test")
    assert result.text  # baseline also generates

    # Restore
    os.environ["USE_RAG"] = "true"
    os.environ["USE_MEMORY"] = "true"
    reload_config()


def test_world_kb_loads():
    """The world JSON loads; the index is non-empty."""
    from retrieval.world_kb import WorldKB
    kb = WorldKB()
    assert len(kb.store) > 0
    hits = kb.retrieve("control room", k=2)
    assert len(hits) > 0


def test_memory_summary_triggers():
    """Once the turn count hits the threshold, summarization is triggered."""
    from memory.short_term import ShortTermMemory
    from memory.summarizer import Summarizer
    from config import CONFIG

    mem = ShortTermMemory()
    summ = Summarizer()
    for i in range(CONFIG.SUMMARY_TRIGGER_TURNS):
        mem.add_turn(f"Player input {i}", f"DM reply {i}")

    assert mem.should_trigger_summary()
    old = mem.pop_oldest_for_summary()
    assert len(old) > 0
    summ.update(old)
    assert summ.get_summary()  # mock returns a fixed summary
