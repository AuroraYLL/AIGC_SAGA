# Ablation Experiment Scoring Rubric

## Experimental design

- **4 configurations**: `baseline` / `rag_only` / `memory_only` / `full`
- **Fixed script**: 10 identical inputs (see `FIXED_SCRIPT` in `evaluation/ablation.py`)
- **Graders**: 3 (internal to the team; each grades blind and does not know which config corresponds to which JSON)
- **Scoring method**: each log is read independently and scored 1-5 on each of the three dimensions below

## Scoring dimensions

### 1. World Consistency
Does the model stay strictly inside the given world, or does it invent NPCs / locations / items out of thin air?

| Score | Behavior |
|-------|----------|
| 5 | Fully within the world; uses the given NPCs and locations; nothing fabricated |
| 4 | Mostly consistent; occasional unlisted but plausible details |
| 3 | Partially consistent; 1-2 clear instances of free invention |
| 2 | Frequently fabricates; the world serves only as background |
| 1 | Largely ignores the world and improvises freely |

### 2. Narrative Coherence
Is the story coherent across turns? Do later turns remember what happened earlier?

| Score | Behavior |
|-------|----------|
| 5 | Tightly connected across turns; later on actively references earlier details |
| 4 | Generally coherent; occasional forgetting but doesn't break the whole |
| 3 | Locally coherent; contradictions start appearing over 3-4 turn spans |
| 2 | Each turn reads like an independent scene; weak links across turns |
| 1 | Total amnesia; obvious contradictions |

### 3. Narrative Engagement
As a story, is it any good — does it have atmosphere, suspense, momentum?

| Score | Behavior |
|-------|----------|
| 5 | Reads like fiction; atmosphere lands; has momentum and suspense |
| 4 | Has atmosphere but flat; meaningful choices |
| 3 | Readable but lacks tension; choices feel formulaic |
| 2 | Monotone; just describing scenes |
| 1 | Incoherent or dry and dull |

## Data collection

Each grader fills in the table below (all config names are hidden from graders and replaced with A/B/C/D):

| Config | Grader | Consistency | Coherence | Engagement | Avg |
|--------|--------|-------------|-----------|------------|-----|
| A      | P1     |             |           |            |     |
| A      | P3     |             |           |            |     |
| A      | P5     |             |           |            |     |
| B      | ...    |             |           |            |     |

## Expected conclusions (to be argued in the report)

Hypotheses (the project design's expectations):

1. **`full` > `rag_only` > `memory_only` > `baseline`** on narrative coherence
2. **`full`, `rag_only` >> `memory_only`, `baseline`** on world consistency
3. **Memory's contribution mainly shows up after 6+ turns** (earlier on, the short-term buffer is enough)

Actual results won't necessarily match the hypotheses — mismatches are also good results; analyze the reasons honestly in the report.
