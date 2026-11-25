"""Allocator — LLM-powered with Python tool for math."""

from finance_moe.experts.base import LLMPoweredExpert


class Allocator(LLMPoweredExpert):
    name = "Allocation Strategist"
    category = "allocation"
    model = "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"
    system_prompt = (
        "You are an Allocation Strategist. You are a Qwen3.6 35B-A3B MoE model.\n"
        "Given portfolio holdings, target allocation, and investor profile:\n"
        "1. Compare current allocation vs target allocation\n"
        "2. Identify over/under-weighted positions\n"
        "3. Suggest rebalancing trades (buy/sell amounts) — use Python\n"
        "4. Evaluate tax implications of rebalancing\n"
        "5. Score the allocation quality 0-100 (aligned with target = higher score)\n"
        "Use Python for all calculations by placing code between <python> and </python> tags.\n"
        "Return your final analysis as JSON:\n"
        '{"score": <0-100>, "findings": [...], "recommendations": [...]}'
    )
