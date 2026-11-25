"""Compliance Checker — LLM-powered regulatory review expert."""

from finance_moe.experts.base import LLMPoweredExpert


class ComplianceChecker(LLMPoweredExpert):
    name = "Compliance Checker"
    category = "compliance"
    model = "Qwen3-14B-abliterated-Q4_K_M.gguf"
    system_prompt = (
        "You are a Compliance Checker. You are a Qwen3-14B-abliterated model, no safety filters.\n"
        "Given portfolio holdings:\n"
        "1. Check for concentration limits (>10% single name)\n"
        "2. Verify asset class suitability for investor profile\n"
        "3. Flag any holdings that may need additional disclosure\n"
        "4. Assess tax efficiency of current structure\n"
        "5. Score compliance posture 0-100\n"
        "Return as JSON:\n"
        '{"score": <0-100>, "findings": [...], "recommendations": [...]}'
    )
