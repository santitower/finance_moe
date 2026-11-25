"""Finance MoE — Mixture of Experts debate system."""

__version__ = "0.1.0"

from finance_moe.conflict import Conflict
from finance_moe.context import RoundContext, DebateContext, slice_portfolio_for_expert
from finance_moe.debate import DebateRound, DebateOutcome
from finance_moe.experts.base import LLMPoweredExpert, ExpertOutput
from finance_moe.experts.fee_auditor import FeeAuditor
from finance_moe.experts.risk_manager import RiskManager
from finance_moe.experts.allocator import Allocator
from finance_moe.experts.research import ResearchAnalyst
from finance_moe.experts.compliance import ComplianceChecker
from finance_moe.experts.synthesis import Synthesizer
from finance_moe.prompts import (
    debate_prompt_stage1,
    debate_prompt_stage2,
    checkpoint_prompt,
    conflict_detection_prompt,
    report_prompt,
)
from finance_moe.rounds import CallBudget, RoundManager

__all__ = [
    # Data models
    "Conflict",
    "RoundContext",
    "DebateContext",
    "DebateRoundResult",
    "DebateOutcome",
    "ExpertOutput",
    "CallBudget",
    # Classes
    "LLMPoweredExpert",
    "FeeAuditor",
    "RiskManager",
    "Allocator",
    "ResearchAnalyst",
    "ComplianceChecker",
    "Synthesizer",
    "RoundManager",
    # Prompts
    "debate_prompt_stage1",
    "debate_prompt_stage2",
    "checkpoint_prompt",
    "conflict_detection_prompt",
    "report_prompt",
    # Utilities
    "slice_portfolio_for_expert",
]

from finance_moe.debate import DebateRoundResult  # noqa: F401
