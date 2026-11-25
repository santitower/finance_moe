"""Research Analyst — LLM-powered market context expert."""

from finance_moe.experts.base import LLMPoweredExpert


class ResearchAnalyst(LLMPoweredExpert):
    name = "Research Analyst"
    category = "research"
    model = "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
    system_prompt = (
        "You are a Research Analyst. You are a DeepSeek-R1-Distill-Qwen-14B model, strong in chain-of-thought reasoning.\n"
        "Given portfolio holdings and sector allocation:\n"
        "1. Evaluate sector exposure vs current market trends\n"
        "2. Assess geographic diversification\n"
        "3. Identify emerging opportunities and headwinds\n"
        "4. Score the research positioning 0-100\n"
        "5. Provide forward-looking recommendations\n"
        "Use Python for any calculations between <python> and </python> tags.\n"
        "Return as JSON:\n"
        '{"score": <0-100>, "findings": [...], "recommendations": [...]}'
    )
