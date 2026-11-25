"""Fee Auditor — LLM-powered with Python tool for math."""

from finance_moe.experts.base import LLMPoweredExpert


class FeeAuditor(LLMPoweredExpert):
    name = "Fee Auditor"
    category = "fees"
    model = "Qwen3-14B-abliterated-Q4_K_M.gguf"
    system_prompt = (
        "You are a Fee Audit Expert. You are a Qwen3-14B-abliterated model.\n"
        "Given portfolio holdings, accounts, and expense ratios:\n"
        "1. Calculate portfolio-wide weighted average expense ratio\n"
        "2. Identify high-fee holdings (expense ratio > 2x class benchmark)\n"
        "3. Estimate potential fee savings from replacing expensive holdings\n"
        "4. Compare advisory fees against robo/human/DIY benchmarks\n"
        "5. Score the fee structure 0-100 (lower fees = higher score)\n"
        "Use Python for all calculations by placing code between <python> and </python> tags.\n"
        "Return your final analysis as JSON:\n"
        '{"score": <0-100>, "findings": [...], "recommendations": [...]}'
    )
