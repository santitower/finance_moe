# Finance Mixture of Experts

A hybrid expert system: Python-backed quantitative analysis + LLM narrative synthesis, with rule-based consensus.

## Architecture

| Expert | Type | Method |
|--------|------|--------|
| Fee Auditor | Python | Expense ratio analysis, swap spread detection |
| Risk Manager | Python | Volatility, Sharpe, max drawdown, concentration |
| Allocator | Python | Concentration, drift, diversification score |
| Synthesizer | LLM (Qwen3.6-35B) | Narrative summary, priority ranking, action items |

Consensus = quantitative outputs aggregated + LLM explanation of conflicts.

## Quick Start

```bash
cd finance_moe
source .venv/bin/activate
python run.py data/sample_portfolio.json
```

## Design Principles

1. **Quantitative experts use math, not LLMs** — allocation, risk, fees are numerical
2. **LLM only for interpretation** — synthesis, narrative, prioritization
3. **Transparent output** — every expert's full analysis included
4. **Local-first** — Qwen3.6-35B at 127.0.0.1:18434, no cloud APIs
