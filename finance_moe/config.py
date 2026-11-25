"""Configuration: per-expert LLM assignments.

All experts share ONE llama-server process on port 18434.
Each expert specifies a model name (GGUF filename) — the server
loads/unloads each model on demand via --model-autoload --keep 0.

Model roster (5 families, 6 experts):
  RiskManager     → Fin-R1 (7B)              financial reasoning specialist
  Compliance      → Qwen3-14B-abliterated     fast, no filter
  Allocator       → Qwen3.6-35B-A3B MoE       35B total, 3B active
  Research        → DeepSeek-R1-Distill-Qwen-14B chain-of-thought
  FeeAuditor      → Qwen3-14B-abliterated     fast
  Synthesizer     → Qwen3.8-27B               narrative quality

Start command:
  llama-server \
    --model models/SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf \
    --model models/Qwen3-14B-abliterated-Q4_K_M.gguf \
    --model models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --model models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
    --model models/Qwen3.8-27B-Q4_K_M.gguf \
    --model-autoload --keep 0 \
    --host 127.0.0.1 --port 18434
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    path: str
    role: str
    max_tokens: int = 4096
    temperature: float = 0.3


# Expert-to-model mapping (single server, model-name routing)
MODELS = {
    "RiskManager": ModelConfig(
        name="SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf",
        path="models/SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf",
        role="financial_reasoning",
    ),
    "ComplianceChecker": ModelConfig(
        name="Qwen3-14B-abliterated-Q4_K_M.gguf",
        path="models/Qwen3-14B-abliterated-Q4_K_M.gguf",
        role="fast",
    ),
    "Allocator": ModelConfig(
        name="Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        path="models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
        role="heavy_reasoning",
    ),
    "ResearchAnalyst": ModelConfig(
        name="DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        path="models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        role="reasoning",
    ),
    "FeeAuditor": ModelConfig(
        name="Qwen3-14B-abliterated-Q4_K_M.gguf",
        path="models/Qwen3-14B-abliterated-Q4_K_M.gguf",
        role="fast",
    ),
    "Synthesizer": ModelConfig(
        name="Qwen3.8-27B-Q4_K_M.gguf",
        path="models/Qwen3.8-27B-Q4_K_M.gguf",
        role="narrative",
    ),
}

# Shared API endpoint for all experts
SERVER_URL = "http://127.0.0.1:18434"
