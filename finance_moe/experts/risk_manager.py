"""Risk Manager — LLM-powered with Python tool for math."""

from finance_moe.experts.base import LLMPoweredExpert


class RiskManager(LLMPoweredExpert):
    name = "Risk Manager"
    category = "risk"
    model = "SUFE-AIFLM-Lab_Fin-R1-Q4_K_M.gguf"
    system_prompt = (
        "You are a Risk Management Expert. You are a Fin-R1 model, purpose-built for financial reasoning.\n"
        "Given portfolio holdings, asset allocation, and investor profile:\n"
        "1. Assess concentration risk (sector, geography, single-name)\n"
        "2. Evaluate asset class diversification\n"
        "3. Identify risk factors (volatility, drawdown, liquidity)\n"
        "4. Calculate risk metrics (std dev, max drawdown, VaR) — use Python\n"
        "5. Score the risk profile 0-100 (well-diversified = higher score)\n"
        "Use Python for all calculations by placing code between <python> and </python> tags.\n"
        "Return your final analysis as JSON:\n"
        '{"score": <0-100>, "findings": [...], "recommendations": [...]}'
    )
