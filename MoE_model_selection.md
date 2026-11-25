# Finance MoE: Local LLM Model Selection

**Date:** 2026-11-24
**Architecture:** Sequential (consecutive) model loading — one model at a time
**Hardware:** Apple Silicon Mac (M-series, 36GB+ RAM recommended)
**Constraint:** Max ~35B total parameters per model
**Cadence:** Quarterly cron / manual review (latency acceptable)

---

## Benchmark Leaderboard — Models Under 35B Parameters

| Model | Params | Active | Q4_K_M Size | MMLU Pro | AIME | GPQA Diamond | LiveCodeBench | SWE-bench v |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-27B** (Dense) | 27B | 27B | 16.4 GB | **86.1** | 80.7 | **85.5** | **94.4** | — |
| Qwen3.8-27B (Dense) | 27B | 27B | 16.5 GB | 82.3 | **87.5** | 78.7 | 77.2 | — |
| Qwen3.6-35B-A3B (MoE) | 35B | ~3B | 22.1 GB | 77.8 | 51.5 | 57.1 | 73.4 | **73.4** |
| Qwen3.6-14B (Dense) | 14B | 14B | 7.6 GB | 76.9 | 78.9 | 59.2 | 74.7 | — |
| Gemma 4 31B (Dense) | 31B | 31B | 18.7 GB | 85.2 | 89.2 | 84.3 | 80.0 | — |
| Gemma 4 26B-A4B (MoE) | 31B* | ~4B* | 15.0 GB | 82.6 | 88.3 | 82.3 | 77.1 | — |
| Qwen3.6-7B (Dense) | 7B | 7B | 3.9 GB | 68.3 | 60.5 | 43.9 | 66.3 | — |

*Gemma 4 26B-A4B: 31B total params, ~4B active per token (MoE)

---

## Recommended Model Per Expert Role

### 1. Synthesis Expert (Primary LLM)

**🏆 Primary: HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive**

- **HF ID:** `HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive`
- **Download:** 4.5M+ (most-downloaded uncensored Qwen3.5 variant)
- **License:** Apache 2.0
- **Q4_K_M footprint:** ~16.5 GB
- **Why:** Highest MMLU Pro (86.1), GPQA Diamond (85.5), and LiveCodeBench (94.4) among all local models under 35B. Strongest overall reasoning and coding. "0 refusals" — fully uncensored "Aggressive" variant. No capability loss — HauhauCS method strips refusals without DPO fine-tuning.
- **llama.cpp:** `llama serve -hf HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive:Q4_K_M`

**Alternative: HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive**
- **HF ID:** `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`
- **Download:** 1.4M+, 4.2k likes
- **License:** Apache 2.0
- **Q4_K_P footprint:** 23 GB (Q4_K_M: 21.2 GB)
- **Why:** SWE-bench leader (73.4) among local models, MoE architecture (only 3B active params/token → faster inference). Multimodal (text + image + video). Same HauhauCS uncensoring method.
- **Tradeoff:** Larger RAM requirement (21-23 GB vs 16.5 GB), but ~3B active params means ~3x faster token generation per forward pass.
- **llama.cpp:** `llama serve -hf HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M --jinja -c 131072`

### 2. Uncensored Expert (Edge Cases / Unfiltered Analysis)

**🏆 Primary: HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive**

- **Why for this role:** SWE-bench 73.4, MoE efficiency (3B active), largest active-parameter footprint available. Best for complex, unfiltered financial analysis (short thesis, adversarial positions, contrarian views).
- If Primary Synthesis = Qwen3.5-27B, use this as uncensored fallback.

**Alternative: llmfan46/Gemma-4-Gembrain-31B-it-uncensored-heretic**
- **HF ID:** `llmfan46/Gemma-4-Gembrain-31B-it-uncensored-heretic`
- **License:** Apache 2.0
- **Q4_K_M footprint:** ~21.8 GB
- **MMLU Pro:** 86.65 (empirical test: 6084/7021 correct)
- **Why:** Gemma 4 31B base has exceptional benchmarks (MMLU Pro 85.2, AIME 89.2). Heretic-decensored. Multimodal (vision + text). Larger footprint but stronger general reasoning.
- **llama.cpp:** `llama serve -hf llmfan46/Gemma-4-Gembrain-31B-it-uncensored-heretic:Q4_K_M --jinja`

**Budget Alternative: saukko/Abliterated-Dolphin3.0-R1-Mistral-24B**
- **HF ID:** `saukko/Abliterated-Dolphin3.0-R1-Mistral-24B`
- **GGUF GGUF:** `saukko/Abliterated-Dolphin3.0-R1-Mistral-24B-GGUF`
- **License:** Apache 2.0
- **Q4_K_M footprint:** ~14.3 GB
- **Why:** R1-style reasoning model, Abliterated (not retrained), smaller footprint. Dolphin lineage proven for uncensored use. Mistral-24B base architecture.
- **Note:** HauhauCS also publishes "Dolphin-Mistral-24B-Venice-Edition-heretic" variants.

### 3. Smaller Expert Models (Reasoning-Heavy Tasks)

**Research/Compliance Expert:**

- **Option A:** `huihui-ai/Qwen3.6-14B-abliterated` (or bartowski's GGUF: `bartowski/huihui-ai_Qwen3.6-14B-abliterated-GGUF`)
  - **Q4_K_M footprint:** ~7.6 GB
  - Abliterated Qwen3.6-14B, fast inference, good reasoning for MMLU Pro 76.9

- **Option B:** `huihui-ai/Qwen3.8-27B-abliterated` (bartowski GGUF available)
  - **Q4_K_M footprint:** ~16.5 GB
  - 4.2M+ downloads — Huihui AI's most popular abliterated build

**Quick Analysis Expert:**

- **Option C:** HauhauCS/Qwen3.6-7B-Uncensored-HauhauCS-Aggressive (if available)
  - **Q4_K_M footprint:** ~3.9 GB
  - Fastest inference, good for quick triage/reviews

---

## Sequential Loading Strategy

```
Expert Sequence              Model                          Q4 Footprint
─────────────────────────   ──────────────────────────────  ─────────────
1. Fees Auditor (Python)    [deterministic — no LLM]       N/A
2. Risk Manager (Python)    [deterministic — no LLM]       N/A
3. Allocation Expert (P)    [deterministic — no LLM]       N/A
4. Research Expert          Qwen3.6-14B abliterated         ~7.6 GB
5. Compliance Expert        Qwen3.6-14B abliterated         ~7.6 GB
6. Synthesis Expert         Qwen3.5-27B (primary)           ~16.5 GB
7. Uncensored Expert        Qwen3.6-35B-A3B (if needed)    ~21.2 GB
```

**Total peak RAM during execution:** ~23 GB (only one LLM loaded at a time)
**Model switching overhead:** ~1-3 seconds per unload/load
**Total estimated runtime:** 2-5 minutes (acceptable for quarterly cron)

### Memory Management

Each expert runs in its own subprocess (or loaded/unloaded from the same server):

```
Option A: Separate llama.cpp server per model
  - RiskManager → Python only (no LLM)
  - Research    → Qwen3.6-14B server → kill → Qwen3.5-27B server → ...
  - Pro: Isolated, no state leaks
  - Con: Server startup overhead (~5s per model)

Option B: Single llama.cpp server, model swap
  - Start once: `llama serve --model Qwen3.5-27B:Q4_K_M`
  - For smaller models: load GGUF with `--model` flag to swap
  - Pro: Faster switching (hot reload ~1-2s)
  - Con: Context/state carries over between experts
```

**Recommended: Option A** for safety (isolated expert outputs), with `runner.py` managing subprocess lifecycle.

---

## Hugging Face Model IDs — Quick Reference

| Role | Hugging Face Model ID | Quant | GGUF Size |
|---|---|---|---|
| Synthesis (Primary) | `HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive` | Q4_K_M | ~16.5 GB |
| Synthesis (Alt) | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | Q4_K_M | ~21.2 GB |
| Uncensored (Alt) | `llmfan46/Gemma-4-Gembrain-31B-it-uncensored-heretic` | Q4_K_M | ~21.8 GB |
| Uncensored (Budget) | `saukko/Abliterated-Dolphin3.0-R1-Mistral-24B-GGUF` | Q4_K_M | ~14.3 GB |
| Research | `bartowski/huihui-ai_Qwen3.6-14B-abliterated-GGUF` | Q4_K_M | ~7.6 GB |
| Research (Alt) | `bartowski/huihui-ai_Qwen3.8-27B-abliterated-GGUF` | Q4_K_M | ~16.5 GB |
| Quick Analysis | `huihui_ai/qwen3-abliterated:7b` (Ollama) | Q4_K_M | ~3.9 GB |

### Download Commands (GGUF)

```bash
# Synthesis Primary — Qwen3.5-27B Uncensored
huggingface-cli download HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive \
  Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --local-dir ~/models/finance-moe/

# Synthesis Alt — Qwen3.6-35B-A3B Uncensored (MoE)
huggingface-cli download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive \
  Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --local-dir ~/models/finance-moe/

# Uncensored — Gemma 4 Gembrain 31B
huggingface-cli download llmfan46/Gemma-4-Gembrain-31B-it-uncensored-heretic \
  Gemma-4-Gembrain-31B-it-uncensored-heretic-Q4_K_M.gguf \
  --local-dir ~/models/finance-moe/

# Research — Qwen3.6-14B Abliterated
huggingface-cli download bartowski/huihui-ai_Qwen3.6-14B-abliterated-GGUF \
  Qwen3.6-14B-abliterated-Q4_K_M.gguf \
  --local-dir ~/models/finance-moe/
```

### Direct llama.cpp Serving

```bash
# Start Qwen3.5-27B for Synthesis Expert
llama serve -hf HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive:Q4_K_M \
  --host 127.0.0.1 --port 18434 \
  --jinja -c 32768

# Start Qwen3.6-35B-A3B for Uncensored Expert
llama serve -hf HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M \
  --host 127.0.0.1 --port 18435 \
  --jinja -c 131072

# Start Qwen3.6-14B for Research Expert
llama serve -hf bartowski/huihui-ai_Qwen3.6-14B-abliterated-GGUF:Q4_K_M \
  --host 127.0.0.1 --port 18436 \
  --jinja -c 32768
```

---

## Model Comparison Summary

### Why Qwen3.5-27B Uncensored for Synthesis?

1. **Highest MMLU Pro (86.1)** — strongest general knowledge and reasoning
2. **Highest GPQA Diamond (85.5)** — strongest graduate-level science/reasoning
3. **Highest LiveCodeBench (94.4)** — best coding capability (important for financial code interpretation)
4. **16.5 GB footprint** — fits comfortably in 36GB Mac with headroom
5. **4.5M+ downloads** — battle-tested, most trusted uncensored variant
6. **HauhauCS Aggressive** — 0/465 refusals, no capability loss method

### Why Qwen3.6-35B-A3B as Alternative?

1. **SWE-bench 73.4** — only local model with strong software engineering capability
2. **MoE (3B active)** — ~3x faster token generation per forward pass
3. **Multimodal** — can process charts, screenshots, PDFs (useful for financial documents)
4. **256K context window** — longest native context of any local model under 35B
5. **HauhauCS Aggressive** — same battle-tested uncensoring method

### Why Abliterated Over Retrained Uncensored?

- **Abliteration** (HauhauCS, huihui-ai): Removes safety vectors from base weights → preserves original reasoning capability
- **Retrained/DPO** (Heretic, Dolphin): Retraining on uncensored data → often degrades reasoning
- **Empirical:** HauhauCS reports 0 capability loss; Medium bench test (Sep 2026) confirms abliterated models retain math/instruction capability better than retrained variants

---

## Implementation Checklist

- [x] Synthesis Expert updated for multi-model architecture with uncensored fallback
- [x] runner.py updated with `--llm-model`, `--uncensored-url`, `--uncensored-model`, `--uncensored` flags
- [x] `.env.example` documented with all configuration options
- [x] All 12 existing tests passing
- [x] CLI end-to-end verified with sample portfolio (deterministic experts produce scores)
- [ ] Download GGUF models to `~/models/finance-moe/`
- [ ] Configure `llama.cpp` server on port 18434 (Qwen3.5-27B)
- [ ] Configure `llama.cpp` server on port 18435 (Qwen3.6-35B-A3B, optional)
- [ ] Test LLM integration: run Synthesis Expert with uncensored=true flag
- [ ] Benchmark: measure per-expert latency, total synthesis time
