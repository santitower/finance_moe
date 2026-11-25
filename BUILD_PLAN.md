# Finance MoE — Complete Build Plan (Refined with Debate Protocol)

## Model Roster (5 families, 6 experts)

All 6 experts share **one llama-server process** on a single port (18434). All GGUF files are listed in `--model` and the `--model-autoload --keep 0` flags load/unload each model on demand. Only one model is loaded in RAM at any time.

| # | Expert | Model Family | GGUF Filename | Size |
|---|--------|-------------|---------------|------|
| 1 | Risk Manager | **Fin-R1** (7B, Shanghai Univ of Finance) | `SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf` | 4.5 GB |
| 2 | Compliance Checker | **Qwen3-14B-abliterated** | `Qwen3-14B-abliterated-Q4_K_M.gguf` | 9.4 GB |
| 3 | Allocator | **Qwen3.6-35B-A3B MoE** (35B total / 3B active) | `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` | 22.3 GB |
| 4 | Research Analyst | **DeepSeek-R1-Distill-Qwen-14B** | `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` | 8.5 GB |
| 5 | Fee Auditor | **Qwen3-14B-abliterated** | `Qwen3-14B-abliterated-Q4_K_M.gguf` | 9.4 GB |
| 6 | Synthesizer | **Qwen3.8-27B** | `Qwen3.8-27B-Q4_K_M.gguf` | 16.5 GB |

### Why these models

**Fin-R1** — Purpose-built for financial reasoning. FinQA: 76.0 (beats Qwen2.5-32B, beats DeepSeek-R1-Distill-Llama-14B). Tiny 7B footprint. Best model in the entire project. Assigned to Risk Manager (most complex financial reasoning).

**Qwen3.6-35B-A3B MoE** — 35B total params, only 3B active per token. Efficient loading, powerful reasoning. Assigned to Allocator (needs math optimization + MoE efficiency).

**DeepSeek-R1-Distill-Qwen-14B** — Strong chain-of-thought reasoning at fast inference speed. Assigned to Research Analyst (market analysis needs reasoning).

**Qwen3.8-27B** — Top-tier general reasoning (MMLU 82.3, AIME 87.5). Assigned to Synthesizer (narrative quality matters most).

**Qwen3-14B-abliterated** — No safety filter, fast inference. Assigned to Fee Auditor and Compliance (fast rule-based checks). Same model file, same server — model name dispatch tells llama-server which file to load.

---

## Build Steps

### Step 1: Download Models (~15-30 min, ~61 GB)

```bash
cd /Users/fsd/finance_moe
chmod +x download_models.sh
bash download_models.sh
```

Or download manually:
```bash
mkdir -p /Users/fsd/finance_moe/models

# 1. Fin-R1 (7B) — Financial domain specialist
huggingface-cli download bartowski/SUFE-AIFLM-Lab_Fin-R1-GGUF \
  --include "SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf" \
  --local-dir /Users/fsd/finance_moe/models

# 2. Qwen3.6-35B-A3B MoE (35B total / 3B active)
huggingface-cli download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF \
  --include "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  --local-dir /Users/fsd/finance_moe/models

# 3. DeepSeek-R1-Distill-Qwen-14B
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF \
  --include "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf" \
  --local-dir /Users/fsd/finance_moe/models

# 4. Qwen3.8-27B
huggingface-cli download bartowski/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-Q4_K_M.gguf" \
  --local-dir /Users/fsd/finance_moe/models

# 5. Qwen3-14B-abliterated
huggingface-cli download bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF \
  --include "huihui-ai_Qwen3-14B-abliterated-Q4_K_M.gguf" \
  --local-dir /Users/fsd/finance_moe/models

# Rename if needed (bartowski uses vendor prefix)
mv models/huihui-ai_Qwen3-14B-abliterated-Q4_K_M.gguf \
   models/Qwen3-14B-abliterated-Q4_K_M.gguf
```

### Step 2: Start Single Llama.cpp Server

```bash
cd /Users/fsd/finance_moe
mkdir -p models

# Start server with all GGUF files — models load/unload on demand
llama-server \
  --model models/SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf \
  --model models/Qwen3-14B-abliterated-Q4_K_M.gguf \
  --model models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
  --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --model-autoload \
  --keep 0 \
  --host 127.0.0.1 \
  --port 18434 &

# Single process. No models loaded until first API call.
# Each API call includes "model" field with the GGUF filename.
# After response, llama-server unloads that model's weights.
# Next call loads the next model's weights from disk.
# Reload latency accepted: 5-30s per disk load.
```

### Step 3: Run Finance MoE

```bash
cd /Users/fsd/finance_moe
.venv/bin/python finance_moe/runner.py data/sample_portfolio.json --format human
```

---

## Architecture — Single Server

```
Portfolio JSON
     │
     ▼
┌──────────────────────────────────────────────────┐
│ runner.py → RoundManager (Python)                │
│   .venv/bin/python finance_moe/runner.py ...     │
│                                                  │
│   Phase 1: 5 sequential API calls to port 18434  │
│     each call specifies model= filename          │
│     llama-server loads/unloads on demand         │
│                                                  │
│   Phase 2: 1 call to Synthesizer (same port)     │
│                                                  │
│   Phase 3: N debate calls (same port,            │
│     reloads between rounds accepted)             │
│                                                  │
│   Phase 4: 1 call to Synthesizer (same port)     │
│                                                  │
│     │                                            │
│     ▼                                            │
│ Executive (resolutions) + Appendix (full         │
│  debate transcripts)                             │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│ llama-server (single process, port 18434)        │
│   --model fin-r1.gguf,compliance.gguf,...        │
│   --model-autoload --keep 0                      │
│                                                  │
│  Request → model="Fin-R1" → loads → runs → ok   │
│  Request → model="Qwen3-14B" → loads → runs     │
│  Request → model="Qwen3.6-35B" → loads → runs   │
│  ... debate rounds: reload between rounds        │
└──────────────────────────────────────────────────┘
```

**Total LLM calls:** 7 + (4 × debate rounds) — capped at 60
**Typical execution:** 12-20 calls (never hits cap)

---

## New Files (Debate System)

| File | Phase | Purpose |
|------|-------|---------|
| `context.py` | Phase 1 | RoundContext, DebateContext (sliding window), data slicing |
| `conflict.py` | Phase 2 | Conflict dataclass |
| `debate.py` | Phase 3 | DebateRound — two-stage debate logic, Synthesizer checkpoints |
| `prompts.py` | Phase 3 | Two-stage prompt templates |
| `rounds.py` | Phase 4 | RoundManager — full pipeline orchestration |

## Modified Files

| File | Changes |
|------|---------|
| `experts/base.py` | Two-stage call method (`stage1` free-text + `stage2` JSON extraction) |
| `experts/synthesis.py` | `detect_conflicts()` (LLM-based), `check_debate_progress()` (checkpoint), `synthesize_report()` |
| `runner.py` | Replace `run_all_experts()` with `RoundManager.execute()`, output formatting |
| `tests/test_experts.py` | Rewrite for debate architecture |

---

## Next Steps

### Build Order (before testing)
1. ✅ All expert modules updated with correct model-name routing
2. ✅ config.py updated with model-to-GGUF filename mapping
3. ✅ Tests rewritten for new architecture (13/13 passing)
4. ✅ BUILD_PLAN.md updated
5. ✅ download_models.sh created
6. ⬜ Download models (`bash download_models.sh`)
7. ⬜ Rename abliterated file if needed
8. ⬜ **DEBATE SYSTEM — NEW**
   - Phase 1: Context engine (`context.py`, `base.py` modifications)
   - Phase 2: Conflict detection (`conflict.py`, `synthesis.py` modifications)
   - Phase 3: Debate protocol — two-stage (`debate.py`, `prompts.py`)
   - Phase 4: Round manager (`rounds.py`)
   - Phase 5: Runner integration (`runner.py` rewrite)
9. ⬜ Start single llama-server with `--model-autoload`
10. ⬜ Run the analyzer with debate system
11. ⬜ Iterate on prompts based on output quality