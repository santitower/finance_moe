"""Tests for Finance MoE debate system.

Tests:
- Import all modules
- Conflict dataclass
- RoundContext / DebateContext
- CallBudget tracking
- ExpertOutput serialization
- DebateRoundResult
- Prompt templates (sanity)
- Portfolio slicing
"""

import json
import os
import sys
import time

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_moe.conflict import Conflict
from finance_moe.context import RoundContext, DebateContext, slice_portfolio_for_expert
from finance_moe.debate import DebateRound, DebateOutcome, DebateRoundResult
from finance_moe.experts.base import LLMPoweredExpert, ExpertOutput
from finance_moe.experts.synthesis import Synthesizer
from finance_moe.prompts import (
    debate_prompt_stage1,
    debate_prompt_stage2,
    checkpoint_prompt,
    conflict_detection_prompt,
    report_prompt,
)
from finance_moe.rounds import CallBudget, RoundManager


# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_PORTFOLIO = {
    "accounts": [
        {
            "type": "401k",
            "balance": 150000,
            "holdings": [
                {"ticker": "AAPL", "sector": "Technology", "shares": 100, "current_price": 195.0},
                {"ticker": "MSFT", "sector": "Technology", "shares": 80, "current_price": 420.0},
                {"ticker": "GOOGL", "sector": "Technology", "shares": 30, "current_price": 175.0},
                {"ticker": "BND", "sector": "Bonds", "shares": 500, "current_price": 72.0},
            ],
        },
        {
            "type": "brokerage",
            "balance": 75000,
            "holdings": [
                {"ticker": "VWO", "sector": "Emerging Markets", "shares": 1000, "current_price": 42.0},
                {"ticker": "VNQ", "sector": "Real Estate", "shares": 200, "current_price": 82.0},
                {"ticker": "JNJ", "sector": "Healthcare", "shares": 50, "current_price": 155.0},
            ],
        },
    ],
    "risk_tolerance": "moderate",
    "time_horizon_years": 20,
    "tax": {
        "marginal_rate": 0.24,
        "state": "CA",
    },
}

FAKE_CONFLICT = Conflict(
    topic="Technology sector concentration",
    expert_a="Risk Manager",
    expert_b="Allocator",
    position_a="55% in tech is dangerously concentrated",
    position_b="Tech dominance is justified for 20-year horizon",
    severity="high",
    data_relevant="Holdings: AAPL 19500, MSFT 33600, GOOGL 5250 = 58350 of 109800 (53%)",
)


# ── Test imports ──────────────────────────────────────────────────────

def test_import_package():
    """All modules import cleanly."""
    import finance_moe
    assert finance_moe.__version__ == "0.1.0"

    from finance_moe import (
        Conflict,
        RoundContext,
        DebateContext,
        CallBudget,
        RoundManager,
        LLMPoweredExpert,
        ExpertOutput,
        Synthesizer,
        FeeAuditor,
        RiskManager,
        Allocator,
        ResearchAnalyst,
        ComplianceChecker,
    )


# ── Test Conflict ─────────────────────────────────────────────────────

def test_conflict_dataclass():
    c = FAKE_CONFLICT
    assert c.topic == "Technology sector concentration"
    assert c.severity == "high"
    assert c.resolved is False
    assert c.rounds_played == 0

    d = c.to_dict()
    assert d["topic"] == c.topic
    assert d["severity"] == c.severity


def test_conflict_defaults():
    c = Conflict(topic="test", expert_a="A", expert_b="B",
                 position_a="x", position_b="y", severity="low")
    assert c.data_relevant == ""
    assert c.resolution == ""
    assert c.unresolved is False


# ── Test RoundContext ─────────────────────────────────────────────────

def test_round_context_creation():
    ctx = RoundContext(portfolio=SAMPLE_PORTFOLIO)
    assert ctx.portfolio is SAMPLE_PORTFOLIO
    assert ctx.round1_results == []
    assert ctx.conflicts == []
    assert ctx.current_expert == ""


def test_round_context_get_summary():
    ctx = RoundContext(portfolio=SAMPLE_PORTFOLIO)
    ctx.round1_results = [
        {
            "expert": "Risk Manager",
            "score": 65,
            "findings": [
                {"metric": "tech_concentration", "value": "53%"},
                {"metric": "bond_allocation", "value": "31%"},
            ],
            "recommendations": ["Reduce tech to 40%"],
        },
    ]
    summary = ctx.get_summary()
    assert "Risk Manager" in summary
    assert "65/100" in summary
    assert "tech_concentration" in summary


def test_round_context_get_conflict_context():
    ctx = RoundContext(portfolio=SAMPLE_PORTFOLIO)
    ctx.round1_results = [
        {
            "expert": "Risk Manager",
            "score": 65,
            "findings": [
                {"metric": "tech_concentration", "value": "53%"},
                {"metric": "bond_allocation", "value": "31%"},
            ],
            "recommendations": ["Reduce tech concentration"],
        },
        {
            "expert": "Allocator",
            "score": 72,
            "findings": [
                {"metric": "tech_growth_outlook", "value": "strong"},
            ],
            "recommendations": ["Maintain tech allocation"],
        },
    ]
    result = ctx.get_conflict_context("Technology sector concentration")
    assert "tech_concentration" in result.lower() or "tech" in result.lower()


# ── Test DebateContext (sliding window) ───────────────────────────────

def test_debate_context_empty():
    dc = DebateContext()
    assert dc.get_prompt_context() == "No previous rounds — first round of this debate."
    assert dc.get_last_rounds() == []


def test_debate_context_add_and_query():
    dc = DebateContext(window_size=3)
    dc.add_round(
        round_num=1,
        expert_a="A",
        expert_b="B",
        stage1_a="reasoning a",
        stage2_a={"position": "A stance", "confidence": 0.8},
        stage1_b="reasoning b",
        stage2_b={"position": "B stance", "confidence": 0.7},
    )

    ctx = dc.get_prompt_context()
    assert "Round 1" in ctx
    assert "A stance" in ctx

    last = dc.get_last_rounds(2)
    assert len(last) == 1


def test_debate_context_summarizes_older():
    dc = DebateContext(window_size=2, summary_window=10)

    for i in range(1, 8):
        dc.add_round(
            round_num=i,
            expert_a="A",
            expert_b="B",
            stage1_a=f"reasoning round {i}",
            stage2_a={"position": f"A round {i}", "confidence": 0.5 + i * 0.05},
            stage1_b=f"reasoning round {i}",
            stage2_b={"position": f"B round {i}", "confidence": 0.5 + i * 0.05},
        )

    ctx = dc.get_prompt_context()
    # Should have full detail for last 2 rounds
    assert "Round 7" in ctx  # most recent
    assert "Round 6" in ctx
    # Should summarize older ones
    assert "Earlier rounds summarized: 5 rounds" in ctx


# ── Test CallBudget ───────────────────────────────────────────────────

def test_call_budget_init():
    budget = CallBudget(total=60)
    assert budget.remaining == 60
    assert budget.total == 60


def test_call_budget_consume():
    budget = CallBudget(total=60)
    assert budget.consume(1) is True
    assert budget.remaining == 59
    assert budget.consume(58) is True
    assert budget.remaining == 1
    assert budget.consume(2) is False  # Not enough
    assert budget.remaining == 1


def test_call_budget_can_afford():
    budget = CallBudget(total=4)
    assert budget.can_afford(4) is True
    budget.consume(4)
    assert budget.can_afford(1) is False


def test_call_budget_summary():
    budget = CallBudget(total=60)
    budget.add_phase_call("phase1", 5)
    budget.add_phase_call("phase3_debate", 12)
    s = budget.get_summary()
    assert "phase1" in s
    assert "budget" in s.lower() or "Budget" in s


# ── Test ExpertOutput ─────────────────────────────────────────────────

def test_expert_output_to_dict():
    eo = ExpertOutput(
        expert_name="Risk Manager",
        category="risk",
        score=65.5,
        findings=[{"metric": "var_95", "value": "-2.1%"}],
        recommendations=["Diversify"],
        confidence=0.85,
        raw={},
    )
    d = eo.to_dict()
    assert d["expert"] == "Risk Manager"
    assert d["score"] == 65.5
    assert d["findings"][0]["metric"] == "var_95"


# ── Test DebateRoundResult ────────────────────────────────────────────

def test_debate_round_result():
    r = DebateRoundResult(
        round_num=1,
        stage1_a="I think tech is overvalued",
        stage2_a={"position": "Underweight tech", "confidence": 0.7},
        stage1_b="Tech is the future",
        stage2_b={"position": "Overweight tech", "confidence": 0.6},
    )
    d = r.to_dict()
    assert d["round_num"] == 1
    assert "overvalued" in d["stage1_a"]


# ── Test DebateOutcome ────────────────────────────────────────────────

def test_debate_outcome_transcript():
    conflict = FAKE_CONFLICT
    outcome = DebateOutcome(
        conflict=conflict,
        transcript=[
            DebateRoundResult(
                round_num=1,
                stage1_a="reasoning a",
                stage2_a={"position": "A stance"},
                stage1_b="reasoning b",
                stage2_b={"position": "B stance"},
            ),
        ],
        resolved=True,
        resolution="Both agreed to rebalance",
        rounds_completed=1,
    )
    text = outcome.get_transcript_text()
    assert "Technology sector concentration" in text
    assert "Round 1" in text
    assert "RESOLVED: Both agreed to rebalance" in text


# ── Test Prompt Templates ─────────────────────────────────────────────

def test_debate_prompt_stage1_returns_two_strings():
    ctx = RoundContext(portfolio=SAMPLE_PORTFOLIO)
    sys_p, user_p = debate_prompt_stage1("Risk Manager", "Allocator", FAKE_CONFLICT, ctx)
    assert "Risk Manager" in sys_p
    assert "Allocator" in user_p
    assert "Stage 1" in sys_p


def test_debate_prompt_stage2_returns_two_strings():
    sys_p, user_p = debate_prompt_stage2(
        "Risk Manager",
        "Tech concentration is too high",
        {"position": "Reduce tech to 40%", "confidence": 0.8},
    )
    assert "Risk Manager" in sys_p
    assert "JSON" in user_p


def test_checkpoint_prompt_returns_two_strings():
    rounds = [
        {
            "round_num": 1,
            "stage2_a": {"position": "A stance 1"},
            "stage2_b": {"position": "B stance 1"},
        },
        {
            "round_num": 2,
            "stage2_a": {"position": "A stance 2"},
            "stage2_b": {"position": "B stance 2"},
        },
    ]
    sys_p, user_p = checkpoint_prompt("A", "B", FAKE_CONFLICT, rounds)
    assert "Moderator" in sys_p or "round 1" in user_p.lower() or "Round 1" in user_p


def test_conflict_detection_prompt_returns_two_strings():
    findings = [
        {"expert": "Risk Manager", "score": 60,
         "findings": [{"metric": "tech", "value": "53%"}],
         "recommendations": ["Reduce tech"]},
        {"expert": "Allocator", "score": 75,
         "findings": [{"metric": "tech_growth", "value": "strong"}],
         "recommendations": ["Maintain tech"]},
    ]
    sys_p, user_p = conflict_detection_prompt(findings, "Portfolio summary")
    assert "Synthesizer" in sys_p
    assert "Risk Manager" in user_p
    assert "conflicts" in user_p


def test_report_prompt_returns_two_strings():
    sys_p, user_p = report_prompt(
        findings=[],
        conflicts=[FAKE_CONFLICT],
        debate_transcripts=[{"round_num": 1}],
        portfolio=SAMPLE_PORTFOLIO,
    )
    assert "Synthesizer" in sys_p
    assert "Executive" in user_p or "report" in user_p.lower()


# ── Test Portfolio Slicing ────────────────────────────────────────────

def test_slice_portfolio_fees():
    result = slice_portfolio_for_expert(SAMPLE_PORTFOLIO, "Fee Auditor")
    assert "holdings" in result
    assert "ticker" in result


def test_slice_portfolio_risk():
    result = slice_portfolio_for_expert(SAMPLE_PORTFOLIO, "Risk Manager")
    assert "risk_tolerance" in result
    assert "time_horizon" in result


def test_slice_portfolio_research():
    result = slice_portfolio_for_expert(SAMPLE_PORTFOLIO, "Research Analyst")
    assert "holdings" in result


def test_slice_portfolio_all_experts():
    experts = ["Fee Auditor", "Risk Manager", "Allocation Strategist",
               "Allocator", "Research Analyst", "Compliance Checker", "Synthesizer"]
    for exp in experts:
        result = slice_portfolio_for_expert(SAMPLE_PORTFOLIO, exp)
        assert isinstance(result, str)
        assert len(result) > 0


# ── Test Synthesizer (no network) ─────────────────────────────────────

def test_synthesizer_class_exists():
    syn = Synthesizer()
    assert syn.model == "Qwen3.8-27B-Q4_K_M.gguf"


# ── Test RunManager (no network) ──────────────────────────────────────

def test_round_manager_init():
    rm = RoundManager()
    assert rm.budget.remaining == 60
    assert rm.round1_results == []
    assert rm.conflicts == []
    assert rm.errors == []
    assert rm.start_time is None


def test_round_manager_progress():
    rm = RoundManager()
    status = rm.get_progress()
    assert "Phase 1" in status
    assert "0/5" in status


# ── Test LLMPoweredExpert base class (no network) ─────────────────────

def test_expert_base_has_required_methods():
    assert hasattr(LLMPoweredExpert, "run")
    assert hasattr(LLMPoweredExpert, "run_with_context")
    assert hasattr(LLMPoweredExpert, "_build_prompt")
    assert hasattr(LLMPoweredExpert, "_chat")
    assert hasattr(LLMPoweredExpert, "_parse_output")


# ── Test Python tool ──────────────────────────────────────────────────

def test_python_tool_basic():
    from finance_moe.experts.base import _python_tool
    result = _python_tool("print(42)")
    assert "42" in result


def test_python_tool_math():
    from finance_moe.experts.base import _python_tool
    result = _python_tool("print(2 + 3 * 4)")
    assert "14" in result


def test_python_tool_error():
    from finance_moe.experts.base import _python_tool
    result = _python_tool("1/0")
    assert "ERROR" in result


# ── Test RunManager._parse_json (integration) ─────────────────────────

def test_synthesizer_json_parsing():
    syn = Synthesizer()

    # Clean JSON
    d = syn._parse_json('{"key": "value"}')
    assert d == {"key": "value"}

    # JSON wrapped in markdown
    d = syn._parse_json("```json\n{\"key\": \"value\"}\n```")
    assert d == {"key": "value"}

    # JSON with surrounding text
    d = syn._parse_json("Here is the data: {\"key\": \"value\"} end.")
    assert d == {"key": "value"}

    # No JSON at all
    d = syn._parse_json("Just plain text")
    assert d == {}


# ── Test full execution path (no network — dry run) ──────────────────

def test_round_manager_calls_no_network():
    """RunManager.execute() should fail gracefully without llama-server.

    It should not crash — it should log errors and return partial results.
    """
    rm = RoundManager(server_url="http://127.0.0.1:18434")

    # This should NOT crash; it should return with errors
    try:
        results = rm.execute(SAMPLE_PORTFOLIO)
        # Should have errors but not crash
        assert "portfolio" in results
        assert isinstance(results.get("errors", []), list)
        assert isinstance(results.get("round1_results", []), list)
        assert isinstance(results.get("budget", ""), str)
    except Exception as e:
        # If it crashes, that's also a failure — we expect graceful degradation
        pytest.fail(f"RoundManager.execute() crashed without network: {e}")


# ── Integration: run.py module ────────────────────────────────────────

def test_runner_format_human():
    """Test that the human formatter handles results without crashing."""
    from finance_moe.runner import format_human

    sample_results = {
        "round1_results": [
            {"expert": "Risk Manager", "score": 65, "findings": [], "recommendations": []},
        ],
        "conflicts": [FAKE_CONFLICT.to_dict()],
        "debate_outcomes": [
            {
                "conflict": {"topic": "tech", "expert_a": "A", "expert_b": "B",
                             "severity": "high", "resolution": "resolved"},
                "transcript": [
                    {"round_num": 1, "stage2_a": {"position": "A"},
                     "stage2_b": {"position": "B"}},
                ],
                "resolved": True,
                "resolution": "agreed",
                "rounds_completed": 1,
            },
        ],
        "report": {"report": "Test report"},
        "errors": [],
        "budget": "Budget: 60/60",
        "elapsed_seconds": 1.0,
    }

    output = format_human(sample_results)
    assert "EXPERT SCORES" in output
    assert "CONFLICTS" in output
    assert "EXECUTION METRICS" in output


def test_runner_format_json():
    """Test that the JSON formatter produces valid JSON."""
    from finance_moe.runner import format_json

    sample_results = {
        "round1_results": [{"expert": "A", "score": 65}],
        "budget": "test budget string",
        "report": {"report": "test"},
        "errors": [],
    }

    output = format_json(sample_results)
    parsed = json.loads(output)
    assert "round1_results" in parsed
    assert parsed["round1_results"][0]["expert"] == "A"


def test_runner_load_portfolio():
    """Test loading the sample portfolio."""
    from finance_moe.runner import load_portfolio
    portfolio = load_portfolio("data/sample_portfolio.json")
    assert "accounts" in portfolio
    assert len(portfolio["accounts"]) >= 2  # At least 2 accounts
    assert any("balance" in acc for acc in portfolio["accounts"])


# ── Test debate transcript format ─────────────────────────────────────

def test_debate_outcome_transcript_empty():
    c = FAKE_CONFLICT
    outcome = DebateOutcome(conflict=c, transcript=[], resolved=False,
                            resolution="", rounds_completed=0)
    text = outcome.get_transcript_text()
    assert "Technology sector concentration" in text
    assert "No rounds" not in text  # should not say "no rounds" if empty list
