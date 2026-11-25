"""Context management — RoundContext, DebateContext (sliding window), data slicing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundContext:
    """Context passed to experts for a given round of analysis."""
    portfolio: dict
    round1_results: list[dict] = field(default_factory=list)  # Phase 1 expert findings
    conflicts: list[Any] = field(default_factory=list)        # Phase 2 Conflict objects
    debate_outcomes: list[dict] = field(default_factory=list)  # Phase 3 results
    call_budget: Any = None                                    # CallBudget instance
    current_expert: str = ""                                   # Who is being called
    previous_experts_done: list[str] = field(default_factory=list)  # Names of completed experts

    def get_summary(self) -> str:
        """Build a textual summary of all Phase 1 findings for prompt injection."""
        parts = []
        for r in self.round1_results:
            expert = r.get("expert", "Unknown")
            score = r.get("score", 0)
            parts.append(f"[{expert}] Score: {score}/100")
            for f in r.get("findings", []):
                metric = f.get("metric", "")
                val = f.get("value", "")
                parts.append(f"  - {metric}: {val}")
            for rec in r.get("recommendations", []):
                parts.append(f"  → {rec}")
            parts.append("")
        return "\n".join(parts)

    def get_conflict_context(self, conflict_topic: str) -> str:
        """Return only the findings relevant to a specific conflict topic."""
        parts = []
        # Extract keywords from topic for flexible matching
        topic_words = set(conflict_topic.lower().split())
        for r in self.round1_results:
            expert = r.get("expert", "Unknown")
            for f in r.get("findings", []):
                text = str(f.get("metric", "")) + " " + str(f.get("value", ""))
                # Check full topic match or any keyword match
                if conflict_topic.lower() in text.lower():
                    parts.append(f"[{expert}] {text}")
                else:
                    for word in topic_words:
                        if len(word) > 3 and word in text.lower():
                            parts.append(f"[{expert}] {text}")
                            break
            if r.get("recommendations"):
                for rec in r.get("recommendations", []):
                    if conflict_topic.lower() in rec.lower():
                        parts.append(f"[{expert}] Rec: {rec}")
        return "\n".join(parts) if parts else "No directly matching findings found."


class DebateContext:
    """Sliding window for debate round history.

    Keeps full detail for the last N rounds, summarizes older ones.
    """

    def __init__(self, window_size: int = 5, summary_window: int = 20):
        """
        Args:
            window_size: Number of recent rounds to keep in full detail.
            summary_window: Total rounds to keep before discarding entirely.
        """
        self.rounds: list[dict] = []
        self.window_size = window_size
        self.summary_window = summary_window

    def add_round(self, round_num: int, expert_a: str, expert_b: str,
                  stage1_a: str, stage2_a: dict, stage1_b: str, stage2_b: dict,
                  checkpoint: Optional[dict] = None) -> None:
        """Append a completed debate round."""
        self.rounds.append({
            "round_num": round_num,
            "expert_a": expert_a,
            "expert_b": expert_b,
            "stage1_a": stage1_a,
            "stage2_a": stage2_a,
            "stage1_b": stage1_b,
            "stage2_b": stage2_b,
            "checkpoint": checkpoint,
        })

    def get_prompt_context(self) -> str:
        """Build the context string for the next debate prompt.

        Full detail for last `window_size` rounds, compressed summaries for older ones.
        """
        if not self.rounds:
            return "No previous rounds — first round of this debate."

        parts = []
        recent = self.rounds[-self.window_size:]
        older = self.rounds[:-self.window_size]

        # Full detail for recent rounds
        for r in recent:
            s2a = json.dumps(r["stage2_a"]) if isinstance(r["stage2_a"], dict) else str(r["stage2_a"])
            s2b = json.dumps(r["stage2_b"]) if isinstance(r["stage2_b"], dict) else str(r["stage2_b"])
            parts.append(
                f"Round {r['round_num']}:\n"
                f"  {r['expert_a']} Stage 1: {r['stage1_a'][:500]}...\n"
                f"  {r['expert_a']} Position: {s2a[:300]}\n"
                f"  {r['expert_b']} Stage 1: {r['stage1_b'][:500]}...\n"
                f"  {r['expert_b']} Position: {s2b[:300]}"
            )
            if r.get("checkpoint"):
                parts.append(f"  Checkpoint: {json.dumps(r['checkpoint'])}")

        # Summaries for older rounds
        if older:
            parts.append(
                f"\n[Earlier rounds summarized: {len(older)} rounds]"
            )
            for r in older[-5:]:  # Show last 5 older round summaries
                s2a = json.dumps(r["stage2_a"]) if isinstance(r["stage2_a"], dict) else str(r["stage2_a"])
                parts.append(
                    f"  Round {r['round_num']}: {r['expert_a']}={s2a[:200]}, "
                    f"{r['expert_b']}={json.dumps(r['stage2_b'])[:200]}"
                )
            if len(older) > 5:
                parts.append(f"  ... and {len(older) - 5} more older rounds omitted.")

        return "\n".join(parts)

    def get_last_rounds(self, n: int = 4) -> list[dict]:
        """Return the last N rounds (for Synthesizer checkpoint)."""
        return self.rounds[-n:]


def slice_portfolio_for_expert(portfolio: dict, expert_role: str,
                                conflict: Any = None) -> str:
    """Pre-compute a relevant data slice for a given expert and optional conflict.

    Args:
        portfolio: Full portfolio dict.
        expert_role: Expert name (e.g. "Fee Auditor").
        conflict: Optional Conflict to narrow the slice.

    Returns:
        Text representation of the relevant portfolio data.
    """
    slices = {
        "Fee Auditor": lambda p: _slice_fees(p),
        "Risk Manager": lambda p: _slice_risk(p),
        "Allocation Strategist": lambda p: _slice_allocation(p),
        "Allocator": lambda p: _slice_allocation(p),
        "Research Analyst": lambda p: _slice_research(p),
        "Compliance Checker": lambda p: _slice_compliance(p),
        "Synthesizer": lambda p: _slice_synthesis(p),
    }

    slicer = slices.get(expert_role, lambda p: json.dumps(p, indent=2))

    result = slicer(portfolio)

    if conflict and expert_role in (conflict.expert_a, conflict.expert_b):
        # Append conflict-relevant data
        result += f"\n\nCONFLICT CONTEXT:\n{conflict.data_relevant}"

    return result


def _slice_fees(p: dict) -> str:
    """Return fee-relevant data."""
    accounts = []
    for acc in p.get("accounts", []):
        info = {"type": acc.get("type"), "balance": acc.get("balance"),
                "expense_ratio": acc.get("expense_ratio")}
        holdings = [{"ticker": h.get("ticker"), "current_price": h.get("current_price"),
                      "shares": h.get("shares")} for h in acc.get("holdings", [])]
        info["holdings"] = holdings
        accounts.append(info)
    return json.dumps({"accounts": accounts}, indent=2)


def _slice_risk(p: dict) -> str:
    """Return risk-relevant data."""
    accounts = []
    for acc in p.get("accounts", []):
        holdings = [{"ticker": h.get("ticker"), "shares": h.get("shares"),
                      "current_price": h.get("current_price"),
                      "sector": h.get("sector")} for h in acc.get("holdings", [])]
        accounts.append({"type": acc.get("type"), "balance": acc.get("balance"),
                         "holdings": holdings})
    return json.dumps({
        "accounts": accounts,
        "risk_tolerance": p.get("risk_tolerance"),
        "time_horizon_years": p.get("time_horizon_years"),
    }, indent=2)


def _slice_allocation(p: dict) -> str:
    """Return allocation-relevant data."""
    return json.dumps({
        "accounts": p.get("accounts", []),
        "total_portfolio_value": p.get("total_portfolio_value", _calc_total(p)),
        "risk_tolerance": p.get("risk_tolerance"),
        "time_horizon_years": p.get("time_horizon_years"),
    }, indent=2)


def _slice_research(p: dict) -> str:
    """Return research-relevant data."""
    holdings = []
    for acc in p.get("accounts", []):
        for h in acc.get("holdings", []):
            holdings.append({"ticker": h.get("ticker"), "sector": h.get("sector"),
                              "current_price": h.get("current_price"),
                              "shares": h.get("shares")})
    return json.dumps({"holdings": holdings}, indent=2)


def _slice_compliance(p: dict) -> str:
    """Return compliance-relevant data."""
    return json.dumps({
        "accounts": p.get("accounts", []),
        "risk_tolerance": p.get("risk_tolerance"),
        "time_horizon_years": p.get("time_horizon_years"),
        "tax": p.get("tax", {}),
    }, indent=2)


def _slice_synthesis(p: dict) -> str:
    """Return full portfolio data for the Synthesizer."""
    return json.dumps(p, indent=2)


def _calc_total(p: dict) -> float:
    return sum(acc.get("balance", 0) for acc in p.get("accounts", []))
