# Finance MoE — Multi-Round Expert Debate Plan

## Core Principles

1. **Debate is always on** — no flag, no shortcut
2. **Unlimited per-conflict debate** — debate continues until convergence or budget exhausted
3. **System-wide cap: 60 total LLM calls** — never reached in practice
4. **Full transparency** — debate transcripts in Appendix of final report
5. **Context-sharing** — each expert sees earlier analysis, not just raw data
6. **LLMs reason, Python orchestrates** — Python provides math/JSON helpers, never makes decisions

## Architecture — Single Server

One llama-server process on port 18434 with `--model-autoload --keep 0`. Models load on demand and unload after each call, regardless of phase. Reload latency (5-30s per disk load) is accepted as a constraint of 32GB RAM.

```
Portfolio JSON
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ llama-server (single process, single port 18434)        │
│   --model fin-r1.gguf,compliance.gguf,allocator.gguf,...│
│   --model-autoload --keep 0                             │
│                                                          │
│  Request → model="fin-r1" → loads → runs → unloads      │
│  Request → model="allocator" → loads → runs → unloads   │
│  ... (Phase 1 completes — 5 disk loads)                  │
│  Request → model="allocator" → reloads from disk        │
│  ... (Phase 3 debate: reloads between rounds accepted)  │
└─────────────────────────────────────────────────────────┘
```

## Debate Protocol

### Round Structure

```
Phase 1: Independent Analysis (5 calls)
  ├── Fee Auditor      → model="Qwen3-14B-abliterated"
  ├── Compliance Checker → model="Qwen3-14B-abliterated" (same file)
  ├── Risk Manager     → model="Fin-R1"
  ├── Research Analyst → model="DeepSeek-R1-14B"
  └── Allocator        → model="Qwen3.6-35B-A3B"

Phase 2: Conflict Detection (1 call)
  └── Synthesizer → model="Qwen3.8-27B" scans all findings → returns list of conflicts

Phase 3: Debate Rounds (N calls, ≤53 remaining)
  For each conflict, debate rounds:
    Step 1A: Free-text reasoning (Expert A responds to Expert B) — thinking ENABLED
    Step 2A: JSON extraction (Expert A summary) — thinking DISABLED
    Step 1B: Free-text reasoning (Expert B responds to Expert A) — thinking ENABLED
    Step 2B: JSON extraction (Expert B summary) — thinking DISABLED
  4 calls per debate round. Synthesizer checkpoint every 4 rounds.

Phase 4: Final Synthesis (1 call)
  └── Synthesizer → writes final report with Executive + Appendix format
```

### Call Budget Tracking

| Call | Phase | Expert | Cost |
|------|-------|--------|------|
| 1-5 | Phase 1 | All 5 experts | 5 |
| 6 | Phase 2 | Synthesizer (detect) | 1 |
| 7-59 | Phase 3 | Debate exchanges | 4 per round (2 stages × 2 experts) |
| 60 | Phase 4 | Synthesizer (report) | 1 |

**Available for debate:** 60 - 7 = 53 calls
**Cost per debate round (1 conflict):** 4 calls (two-stage approach)
**Max debate rounds at 1 conflict:** 13 rounds (plenty)
**Max debate rounds at 2 conflicts:** 6 rounds per conflict
**Max debate rounds at 3 conflicts:** 4 rounds per conflict

### Debate Prompt Template — Two-Stage Approach

**Stage 1A (Expert A reasoning — thinking enabled, free text):**
```
You are the {expert_name}.

CONFLICT: {topic}

Your original analysis: "{your_original_finding}"
{other_expert}'s original analysis: "{their_original_finding}"

Their latest position on this conflict: "{their_latest_position}"

Relevant portfolio data:
{relevant_data}

Think through your response carefully. Challenge their reasoning where you disagree.
Cite specific portfolio data. Explain if their position changed your thinking.
Respond in free text — no JSON format required.
```

**Stage 2A (Expert A JSON extraction — thinking disabled, mechanical):**
```
Based on the reasoning above, extract your position as JSON:
{
  "position": "one-line stance",
  "confidence": 0.0-1.0,
  "key_arguments": ["short argument", "short argument"],
  "data_points": ["number or metric"]
}
```

**Stage 1B / Stage 2B** — same pattern, roles swapped.

### Conflict Detection — Synthesizer-Based (Phase 2)

Phase 2 Synthesizer identifies substantive disagreements (not heuristic score gaps):

```python
# Phase 2: ask Synthesizer to analyze all expert findings
# and return structured conflicts
prompt = f"""You are analyzing findings from 5 financial experts.
Identify any substantive disagreements where experts disagree on facts,
conclusions, or recommendations. Ignore areas where they just score
differently on the same observation.

Findings:
{format_findings(all_findings)}

Portfolio summary:
{summarize_portfolio(portfolio)}

Return JSON:
{{"conflicts": [
  {{
    "topic": "brief description",
    "expert_a": "name",
    "expert_b": "name",
    "position_a": "what expert A thinks",
    "position_b": "what expert B thinks",
    "severity": "high|medium|low",
    "data_relevant": "which portfolio data is at issue"
  }}
]}}
"""
```

### Debate Termination — Synthesizer Checkpoint

Every 4 rounds, a Synthesizer checkpoint evaluates whether debate is progressing:

```python
# After every 4 debate rounds:
if len(self.round_history) % 4 == 0:
    checkpoint = synthesizer.check_debate_progress(
        conflict=self.conflict,
        round_history=self.round_history[-4:]
    )
    if checkpoint["recommendation"] == "stop":
        return True, checkpoint["reason"]  # RESOLVED
```

Synthesizer checkpoint prompt:
```
You're monitoring a debate between {expert_a} and {expert_b}
on: {topic}. Recent rounds:

{round_summaries}

Has either expert changed position? Is debate making real progress?

{{"positions_changed": true/false,
  "making_progress": true/false,
  "recommendation": "continue|stop",
  "reason": "one sentence"}}
```

Also terminates immediately if:
- Both experts output the same position (consensus)
- Budget exhausted (60 total calls)

## Output Format

```
==============================================================
FINANCE MoE ANALYSIS — 2024-01-15T14:30:00Z
==============================================================

Portfolio Value: $500,000.00

--- Expert Scores ---
  [████████░░] Fee Auditor: 85/100
  [██████░░░░] Compliance Checker: 60/100
  [███░░░░░░░] Risk Manager: 30/100
  [███████░░░] Research Analyst: 70/100
  [████████░░] Allocator: 75/100

--- Expert Findings ---
[Fee Auditor] Expense ratios: 0.35% avg (reasonable)
[Compliance] Tech concentration: 75% (exceeds 10% single-name warning)
[Risk Manager] Sector concentration: EXTREME — 75% in technology
[Research Analyst] Tech sector outlook: Strong (AI, cloud growth)
[Allocator] Sector tilt: INTENTIONAL — 50% target, not 75%

--- Conflict Resolutions ---
CONFLICT 1: Tech Concentration (HIGH)
  Participants: Risk Manager ↔ Allocator
  Resolution: Both agree 75% is excessive. Recommend reduce to 50%.

CONFLICT 2: Cash Position (MEDIUM)
  Participants: Risk Manager ↔ Research Analyst
  Resolution: Research concedes 5-10% cash buffer is reasonable.
  Recommend reduce from 25% to 5-10%.

==============================================================
FINAL RECOMMENDATIONS
==============================================================
1. Reduce tech concentration from 75% to 50% (Risk ↔ Allocator)
2. Reduce cash from 25% to 5-10% (Risk ↔ Research)
3. No action on expense ratios

Overall Health Score: 55/100

==============================================================
APPENDIX: FULL DEBATE TRANSCRIPTS
==============================================================

CONFLICT 1: Tech Concentration
──────────────────────────────────────────────────────────────
Participants: Risk Manager ↔ Allocator

Round 1, Stage 1 (Risk Manager):
  "Even intentional concentration has limits. The investor profile
   shows moderate risk tolerance. 75% tech violates all
   diversification guidelines and concentration risk exceeds
   the portfolio's risk budget by 50%."

Round 1, Stage 2 (Risk Manager JSON):
  {"position": "Reduce from 75% to 50%", "confidence": 0.9,
   "key_arguments": ["Exceeds diversification guidelines", "50% above risk budget"],
   "data_points": ["75% vs 50% target", "concentration risk >50%"]}

Round 1, Stage 1 (Allocator):
  "I agree 75% is too high. My target was 50%, not 75%.
   The current allocation should be trimmed."

... (continues for all rounds)

==============================================================
```

---

## File Architecture

```
finance_moe/
├── __init__.py
├── config.py              # Model aliases mapping (expert → GGUF path)
├── context.py             # RoundContext, sliding window, data slicing
├── conflict.py            # Conflict dataclass
├── debate.py              # DebateRound, Synthesizer checkpoint logic
├── prompts.py             # Two-stage prompt templates
├── rounds.py              # RoundManager — full pipeline orchestration
├── runner.py              # CLI entry, output formatting
├── experts/
│   ├── __init__.py
│   ├── base.py            # Expert base, two-stage call method
│   ├── fee_auditor.py
│   ├── risk_manager.py
│   ├── allocator.py
│   ├── research.py
│   ├── compliance.py
│   └── synthesis.py       # detect_conflicts(), check_debate_progress(), synthesize_report()
├── tests/
│   ├── __init__.py
│   └── test_experts.py
├── data/
│   └── sample_portfolio.json
├── download_models.sh
├── BUILD_PLAN.md
└── DEBATE_PLAN.md
```

---

## Phase Plans

### Phase 1: Context Engine (3-4 hours)

**Goal:** Wire up context sharing, data slicing, sliding window

**Create:**
- `finance_moe/context.py`
  - `RoundContext` dataclass: portfolio, round1_results, conflicts, debate_outcomes, call_budget
  - `DebateContext` class: sliding window with round summaries
    - `add_round()` — appends round, auto-compresses older rounds to summaries
    - `get_prompt_context()` — builds context string for LLM prompt
    - Full detail for last 5 rounds, one-line summaries for older rounds
  - `slice_portfolio_for_expert(expert_role, conflict)` — pre-compute per-expert data slices

**Modify:**
- `finance_moe/experts/base.py` — add `context` parameter, `_get_relevant_context_fields()` method

---

### Phase 2: Conflict Detection (2-3 hours)

**Goal:** Synthesizer identifies disagreements between findings

**Modify:**
- `finance_moe/experts/synthesis.py`
  - Add `detect_conflicts(all_findings, portfolio)` — asks Synthesizer to find substantive disagreements
  - Returns list of Conflict dataclasses

**Tests:**
- Conflict detection returns structured conflicts for sample data

---

### Phase 3: Debate Protocol (5-6 hours)

**Goal:** Two-stage structured debate between experts

**Create:**
- `finance_moe/debate.py`
  - `DebateExchange` dataclass: round_num, expert_a (stage1, stage2), expert_b (stage1, stage2)
  - `DebateRound` class:
    - `run()` — executes full debate loop until resolved or budget exhausted
    - `step_a_stage1()` — free-text reasoning (Expert A), thinking ENABLED
    - `step_a_stage2()` — JSON extraction (Expert A), thinking DISABLED
    - `step_b_stage1()` — free-text reasoning (Expert B), thinking ENABLED
    - `step_b_stage2()` — JSON extraction (Expert B), thinking DISABLED
    - `is_resolved()` — check for consensus or Synthesizer checkpoint termination
    - `get_transcript()` — formatted output for Appendix
  - Synthesizer checkpoint: called every 4 rounds

- `finance_moe/prompts.py`
  - `debate_prompt_stage1(expert_a, expert_b, conflict, context)` — free-text reasoning
  - `debate_prompt_stage2(expert, stage1_response)` — JSON extraction
  - `checkpoint_prompt(expert_a, expert_b, conflict, recent_rounds)` — progress evaluation

**Debate flow:**
```python
debate = DebateRound(conflict, expert_a, expert_b, context, budget)
results = debate.run()
# results.debate_transcript — full exchange log (for Appendix)
# results.conflict.resolved — bool
# results.conflict.resolution_summary — one-line outcome
```

---

### Phase 4: Round Manager (3-4 hours)

**Goal:** Orchestrate all phases with call budget tracking

**Create:**
- `finance_moe/rounds.py`
  - `CallBudget` class: tracks total calls, remaining calls, per-expert counts
  - `RoundManager` class:
    - `execute(portfolio)` — runs full pipeline
      - Phase 1: Run 5 experts sequentially (single server, reloads accepted)
      - Phase 2: Synthesizer detects conflicts
      - Phase 3: Run debates for each conflict (with Synthesizer checkpoints)
      - Phase 4: Synthesizer writes final report
    - `get_progress()` — status string for logging
    - `get_results()` — all findings + transcripts + report

**Modify:**
- `finance_moe/runner.py` — replace `run_all_experts()` with `RoundManager.execute()`

---

### Phase 5: Integration & Polish (2-3 hours)

**Goal:** Tie everything together, output formatting, error handling

**Modify:**
- `finance_moe/runner.py`
  - Update output formatting: Executive section (resolutions) + Appendix (full transcripts)
  - Show conflict resolutions inline in Executive section
  - Format expert scores and findings clearly
  - Add progress reporting to stderr

**Error handling:**
- Connection error (llama-server crash/OOM) → log, mark conflict unresolved, move on
- Timeout (>120s) → log, mark call failed, continue with remaining experts
- No retries (local server — transient failures don't self-recover)

---

## Call Budget Safety

**Hard limit: 60 total LLM calls per analysis**

| Stage | Calls | Description |
|-------|-------|-------------|
| Phase 1 | 5 | All 5 experts (independent) |
| Phase 2 | 1 | Synthesizer detects conflicts |
| Phase 3 | 53 max | Debates: 4 calls × rounds (varies by conflict count) |
| Phase 4 | 1 | Synthesizer final report |

**Budget at 4 calls/round:**
- 1 conflict → 13 rounds max
- 2 conflicts → 6 rounds per conflict
- 3 conflicts → 4 rounds per conflict

**Safety mechanisms:**
1. Hard cap at 60 total calls
2. Synthesizer checkpoint every 4 rounds to stop circular debates
3. If budget exhausted, mark conflicts as unresolved and continue to Phase 4
4. Connection/timeout errors → mark expert call failed, move on

**Why 60 is safe:**
- Normal analysis with 1 conflict: ~12 calls (5+1+4+1+1)
- Normal analysis with 2 conflicts: ~16 calls
- Even with 3 conflicts and 5 rounds each: ~28 calls
- 60 is ~2× normal, never reached in practice

## Error Handling Notes

- llama-server runs locally on 127.0.0.1:18434
- No external API — errors are only: connection refused (server crash), timeout (model hanging)
- With `--keep 0` and sequential loading, OOM should not occur (server drops old model before loading new one)
- No retry logic — local server failures don't self-recover; just mark and continue