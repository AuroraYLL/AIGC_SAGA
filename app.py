"""
Gradio UI entry point.
[OWNER: P4 Frontend]

Features:
- Left: main chat area (story + choice buttons)
- Right: status panel (HP / location / items / current summary)
- Top controls: New Game / Generate Final Novel

Day 1 minimal version: chat box only. Other features will be added
progressively during Weeks 1-2.
"""
import gradio as gr

from core.generator import StoryEngine
from retrieval.world_kb import WorldKB
from memory.short_term import ShortTermMemory
from memory.summarizer import Summarizer
from story.compiler import compile_story


# ---- Global engine (good enough for single-user demo; multi-user needs session state) ----
_engine: StoryEngine | None = None
_player_state = {"hp": 100, "location": "Airlock", "items": []}


def _init_engine() -> StoryEngine:
    """Constructed on first call. Takes 5-10 seconds (embedding model load)."""
    global _engine, _player_state
    world_kb = WorldKB()
    memory = ShortTermMemory()
    summarizer = Summarizer()
    _engine = StoryEngine(world_kb, memory, summarizer)
    _player_state = {"hp": 100, "location": "Airlock", "items": []}
    return _engine


def _apply_state_delta(delta: dict):
    """Merge the LLM's returned state changes into the player state."""
    global _player_state
    if "location" in delta and delta["location"]:
        _player_state["location"] = delta["location"]
    if "hp_change" in delta:
        _player_state["hp"] = max(0, min(100, _player_state["hp"] + int(delta.get("hp_change", 0))))
    for item in delta.get("items_gained", []):
        if item not in _player_state["items"]:
            _player_state["items"].append(item)
    for item in delta.get("items_lost", []):
        if item in _player_state["items"]:
            _player_state["items"].remove(item)


def _format_state() -> str:
    return (
        f"**❤️ HP:** {_player_state['hp']}/100\n\n"
        f"**📍 Location:** {_player_state['location']}\n\n"
        f"**🎒 Items:**\n"
        + ("\n".join(f"- {i}" for i in _player_state['items']) if _player_state['items'] else "_(empty)_")
    )


def on_send(user_message: str, chat_history: list):
    """User sends a message → call the engine → append to the chat."""
    global _engine
    if _engine is None:
        _init_engine()

    if not user_message.strip():
        return "", chat_history, _format_state(), gr.update()

    result = _engine.generate(user_message)
    _apply_state_delta(result.state_delta)

    # Build story + choices
    reply = result.text
    if result.choices:
        reply += "\n\n**Available actions:**\n" + "\n".join(f"- {c}" for c in result.choices)

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    # Update choice-button visibility
    btn_updates = [
        gr.update(value=c, visible=True) if i < len(result.choices) else gr.update(visible=False)
        for i, c in enumerate(result.choices + ["", "", ""])
    ][:3]

    return "", chat_history, _format_state(), *btn_updates


def on_choice(choice_text: str, chat_history: list):
    """Clicking a choice button == sending that choice's text."""
    return on_send(choice_text, chat_history)


def on_new_game():
    _init_engine()
    history = [{"role": "assistant", "content": _engine.world_kb.get_opening()}]
    return (
        history,
        _format_state(),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def on_compile():
    """Week 2 feature: generate the final novel."""
    global _engine
    if _engine is None or len(_engine.memory) == 0:
        return "No story recorded yet — play a few turns before generating the novel."
    return compile_story(_engine.get_full_story_log())


# ============ UI ============
with gr.Blocks(title="Deep Space Echo — AI Dungeon", analytics_enabled=False) as demo:
    gr.Markdown("# 🚀 Deep Space Echo — AI Interactive Story")
    gr.Markdown("_AIGC group project · RAG + Memory + LLM_")

    with gr.Row():
        # Left: chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Story",
                type="messages",
                height=520,
                show_copy_button=True,
            )
            with gr.Row():
                choice_btn1 = gr.Button("Choice 1", visible=False, variant="secondary")
                choice_btn2 = gr.Button("Choice 2", visible=False, variant="secondary")
                choice_btn3 = gr.Button("Choice 3", visible=False, variant="secondary")

            with gr.Row():
                input_box = gr.Textbox(
                    placeholder="Type your action (or click a choice button above)",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        # Right: status + controls
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Character Status")
            state_md = gr.Markdown(_format_state())
            gr.Markdown("---")
            new_game_btn = gr.Button("🔄 New Game", variant="secondary")
            compile_btn = gr.Button("📖 Generate Final Novel", variant="secondary")
            gr.Markdown("### 📝 Novel Output")
            novel_md = gr.Markdown("_(Play for a while, then click the button on the left to generate)_")

    # ---- Event bindings ----
    send_btn.click(
        on_send,
        [input_box, chatbot],
        [input_box, chatbot, state_md, choice_btn1, choice_btn2, choice_btn3],
    )
    input_box.submit(
        on_send,
        [input_box, chatbot],
        [input_box, chatbot, state_md, choice_btn1, choice_btn2, choice_btn3],
    )
    for btn in [choice_btn1, choice_btn2, choice_btn3]:
        btn.click(
            on_choice,
            [btn, chatbot],
            [input_box, chatbot, state_md, choice_btn1, choice_btn2, choice_btn3],
        )

    new_game_btn.click(
        on_new_game,
        None,
        [chatbot, state_md, choice_btn1, choice_btn2, choice_btn3],
    )
    compile_btn.click(on_compile, None, novel_md)


if __name__ == "__main__":
    #demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False, inbrowser=True)
    demo.launch(share=True, show_api=False, inbrowser=True)
