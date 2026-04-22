# AI Dungeon — Deep Space Echo 🚀

Interactive AI narrative generation system (AIGC course group project). 3-week timeline, 6-person team.

Combines **RAG (world knowledge) + Memory (story memory) + LLM (generation)** to build an AI storytelling system with long-term coherence.

---

## 🧱 Directory structure & role assignments

```
ai-dungeon/
├── app.py                     # Gradio UI entry point           [P4 Frontend]
├── main.py                    # CLI smoke-test entry point      [Shared]
├── config.py                  # Global config + ablation flags  [P1 Tech Lead]
├── .env.example               # API key template
├── requirements.txt
│
├── core/                      # [P1 Tech Lead] Main pipeline (contract, DO NOT change)
│   ├── generator.py           #   Main pipeline StoryEngine.generate()
│   ├── llm.py                 #   LLM wrapper + MOCK switch
│   └── prompts.py             #   5-section system prompt (core algorithm)
│
├── retrieval/                 # [P2 Retrieval Engineer]
│   ├── embedder.py            #   Local embeddings (bge-small-en)
│   ├── vector_store.py        #   FAISS wrapper
│   └── world_kb.py            #   World KB loading + retrieval
│
├── memory/                    # [P3 Memory Engineer]
│   ├── short_term.py          #   Recent N-turn buffer
│   └── summarizer.py          #   Long-term compression (LLM summary)
│
├── world/                     # [P5 World Designer]
│   └── world_kb.json          #   World data (example: derelict space station)
│
├── story/                     #   Week 2 task
│   └── compiler.py            #   Final novel generation
│
├── evaluation/                # [P6 Evaluation & Report]
│   ├── ablation.py            #   Ablation experiment harness
│   └── rubric.md              #   Scoring criteria
│
└── tests/
    └── test_smoke.py          #   Smoke tests (must pass on every commit)
```

---

## 🔒 Development contract (DO NOT change arbitrarily!)

### 1. Main pipeline function signature (`core/generator.py`)

```python
engine = StoryEngine(world_kb, memory, summarizer)
result = engine.generate(user_input: str) -> GenerateResult
#   result.text: str           — story description
#   result.choices: List[str]  — 2-3 player options
#   result.state_delta: dict   — state changes
```

**All module-facing interfaces are locked; parallel development by 6 people depends on this contract.** Changing any signature requires prior alignment in the group chat.

### 2. Ablation flags (`config.py`)

```python
USE_RAG      = os.getenv("USE_RAG", "true")       # P2's module can be disabled
USE_MEMORY   = os.getenv("USE_MEMORY", "true")    # P3's module can be disabled
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "true")  # Default on during dev, saves money
```

**P6's ablation experiments depend entirely on these flags. No code may hardcode a bypass around these switches.**

### 3. Mock LLM switch

During development, keep `USE_MOCK_LLM=true` (default) — zero API cost. Only switch to the real API for integration tests / demos.

---

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment variables
cp .env.example .env
# Edit .env and add your DeepSeek API key (only needed when using a real LLM)

# 3. Smoke test (uses mock)
python main.py

# 4. Launch UI
python app.py
```

---

## 📅 3-week timeline (excerpt)

- **Week 1**: Day 1 align on contract → Day 3 MVP vertical slice → Day 7 RAG integration
- **Week 2**: Memory compression → JSON output format → team-wide dogfooding → bug fixes
- **Week 3**: Ablation experiments → final novel → report + demo

## ✅ Day 1 must-do checklist

- [ ] Everyone forks the repo and runs `python main.py` (verify mock mode works)
- [ ] P1 confirms final `GenerateResult` fields in the group chat
- [ ] P5 picks the theme (suggested: derelict space station / Cthulhu detective / post-apocalyptic wasteland)
- [ ] P6 posts the scoring rubric to the group
- [ ] Create a 6-person GitHub Projects board with issues split by module

## ⚠️ The easiest mistakes to make

1. **Not locking down `generate()` return fields on Day 1** → Week 2 integration blows up
2. **Someone hardcodes a bypass around `USE_RAG` / `USE_MEMORY` in business code** → Week 3 ablation experiments can't run
3. **Forgetting to keep `USE_MOCK_LLM` enabled during development** → API budget burned in week 1
