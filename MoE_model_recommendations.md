# LLM Model Recommendations for Finance MoE System

## Machine Context
- **Hardware:** macOS MacBook Pro ARM64 (Apple Silicon)
- **Current model:** Qwen3.6-35B-A3B (MoE, 35B total / 3B active)
- **Approx. VRAM:** ~22 GB Q4_K_M for Qwen3.6-35B-A3B
- **Constraint:** Models ≤ Qwen3.6-35B-A3B footprint; local-only; at least one uncensored

---

## Recommended Expert Model Set (Balanced Configuration)

| Expert Role | Model | Params | Q4_K_M Size | Key Strength | License |
|---|---|---|---|---|---|
| **Reasoning/Analysis** | Qwen3-14B | 14B | ~8.9 GB | Best-in-class reasoning at this size; strong math/logic | Apache 2.0 |
| **General Synthesis** | Llama-3.1-8B-Instruct | 8B | ~4.9 GB | Excellent instruction following; broad tooling | Llama 3.1 Community |
| **Coding/Technical** | Qwen3-Coder-7B-Instruct | 7B | ~4.7 GB | Highest code benchmark scores under 10B; HumanEval 65.8+ | Apache 2.0 |
| **Uncensored (Devil's Advocate)** | Qwen3-14B-abliterated | 14B | ~8.9 GB | Refusals removed; uncensored reasoning | Apache 2.0 (based on Qwen3-14B) |
| **Total footprint** | | | **~27 GB** | All fit comfortably | |

### Why These Four?

1. **Architectural diversity:** Qwen3 (MoE), Llama 3.1 (dense), Qwen-Coder (dense, code-specialized) — different architectures produce genuinely different perspectives, not just fine-tune variants of the same model.

2. **Different size tiers:** 7B–14B range gives you fast-response (7B) and deep-reasoning (14B) experts. The 7B coder can do quick code analysis; the 14B reasoning expert handles complex financial logic.

3. **Uncensored option:** Qwen3-14B-abliterated (by huihui-ai on HF) removes alignment/refusal behavior while keeping the base model's reasoning capability. For your finance MoE, this is useful for "devil's advocate" analysis — finding risks, edge cases, or contrarian views that a censored model might hedge away.

4. **All Apache 2.0 or permissive:** Qwen models are Apache 2.0. Llama 3.1 has a community license (free for most use). No API keys needed.

---

## Uncensored Model Options (Under 35B)

| Model | HF Source | Q4_K_M | Notes |
|---|---|---|---|
| **Qwen3-14B-abliterated** | `huihui-ai/Qwen3-14B-abliterated` | ~8.9 GB | Abliteration removes refusal without full retraining. Best reasoning quality among small uncensored models. |
| **Qwen3.6-35B-A3B-abliterated** | `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF` | ~22 GB | Uncensored version of your current model. Slower but highest quality uncensored option. |
| **Dolphin 3.0 Llama 3.1 8B** | `dphn/Dolphin3.0-Llama3.1-8B-GGUF` | ~4.9 GB | Eric Hartford's uncensored Llama fine-tune. Fast, good for rapid analysis rounds. |
| **Dolphin 3.0 R1 Mistral 24B** | `eaddario/Dolphin3.0-R1-Mistral-24B-GGUF` | ~14 GB | Reasoning-enhanced (R1) uncensored. Strongest reasoning among uncensored under 35B. ~14 GB Q4. |
| **Hermes 3 Llama 3.1 8B** | `NousResearch/Hermes-3-Llama-3.1-8B` | ~4.9 GB | Nous' own uncensored model. "Unlocked, uncensored, highly steerable." Good for synthesis tasks. |

**Recommendation:** Use `Qwen3-14B-abliterated` for your uncensored expert. It has the best reasoning quality among 14B uncensored models and fits well in the size budget. If you want stronger reasoning at the cost of RAM, `Dolphin 3.0 R1 Mistral 24B` is the next best uncensored reasoning model.

---

## Alternative Configurations

### Lightweight (All ~10GB total, fits any Mac with 16GB+ RAM)
| Expert | Model | Q4_K_M |
|---|---|---|
| Reasoning | Qwen3-7B-Instruct | ~4.7 GB |
| Synthesis | Llama-3.1-8B-Instruct | ~4.9 GB |
| Coding | Qwen3-Coder-7B-Instruct | ~4.7 GB |
| Uncensored | Dolphin 3.0 Llama 3.1 8B | ~4.9 GB |
| **Total** | | **~19 GB** |

### Maximum Quality (Your current setup + specialists)
| Expert | Model | Q4_K_M |
|---|---|---|
| Reasoning | Qwen3.6-35B-A3B (current) | ~22 GB |
| Synthesis | Qwen3-8B-Instruct | ~5 GB |
| Coding | Qwen3-Coder-14B | ~9 GB |
| Uncensored | Qwen3.6-35B-A3B-abliterated | ~22 GB |
| **Total** | | **~58 GB** (requires swapping, not concurrent) |

---

## Model-Specific Benchmark Highlights

**Qwen3-14B:**
- MMLU: ~75–77% (beats Llama 3.1 8B by ~7 points)
- GSM8K: ~87–91% (strong math for its size)
- Best all-around reasoning model under 15B

**Qwen3-Coder-7B:**
- HumanEval: 65.8+ (Qwen's own numbers)
- Purpose-built for code generation and analysis
- Beats Llama 3.1 8B on code tasks by 10+ points

**Llama-3.1-8B-Instruct:**
- MMLU: ~68–70%
- MT-Bench: ~8.1–8.2
- Best general-purpose 8B; largest ecosystem of fine-tunes and tooling

**Dolphin 3.0 R1 Mistral 24B:**
- R1 = reasoning-capable (chain-of-thought trained)
- Strongest uncensored reasoning model under 35B
- ~14 GB Q4_K_M

---

## Implementation Notes

1. **Sequential expert calls:** Your current runner calls experts sequentially via OpenAI-compatible API. Each model gets its own `.env` model identifier:
   ```
   EXPERT_REASONING_MODEL=Qwen3-14B
   EXPERT_SYNTHESIS_MODEL=Llama-3.1-8B-Instruct
   EXPERT_CODING_MODEL=Qwen3-Coder-7B-Instruct
   EXPERT_UNCENSORED_MODEL=Qwen3-14B-abliterated
   ```

2. **Model loading on Apple Silicon:** Use `llama.cpp` / `ollama` for GGUF-based inference. Each model needs to be downloaded once (~5–22 GB each) and will be cached on disk. Model switching at inference time adds ~1–3 seconds for model reload.

3. **Parallel inference:** Running all 4 models simultaneously requires ~27 GB unified memory. Your current Qwen3.6-35B-A3B already uses ~22 GB. For true parallel expert calls, consider running a single multi-model server or rotating models sequentially (faster on Apple Silicon than parallel loading).

4. **Abliterated vs. fully uncensored:** "Abliterated" models (huihui-ai builds) remove refusal patterns through a surgical process — they keep reasoning quality better than models retrained from scratch. "Dolphin" models are fine-tuned uncensored versions. For your finance use case, abliterated models are preferable because they maintain analytical rigor.
