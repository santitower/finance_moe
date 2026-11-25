"""RoundManager — full pipeline orchestration with call budget tracking."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from finance_moe.conflict import Conflict
from finance_moe.context import RoundContext
from finance_moe.debate import DebateOutcome, DebateRound
from finance_moe.prompts import (
    conflict_detection_prompt,
    report_prompt,
)


@dataclass
class CallBudget:
    """Track total and per-expert LLM call counts."""
    total: int = 60
    remaining: int = 60
    per_expert: dict[str, int] = field(default_factory=dict)
    phase_calls: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.remaining == 60 and self.total != 60:
            self.remaining = self.total

    def consume(self, n: int = 1) -> bool:
        """Consume N calls. Returns True if sufficient budget remains."""
        if self.remaining >= n:
            self.remaining -= n
            return True
        return False

    def can_afford(self, n: int = 1) -> bool:
        return self.remaining >= n

    def add_phase_call(self, phase: str, count: int = 1) -> None:
        self.phase_calls[phase] = self.phase_calls.get(phase, 0) + count

    def get_summary(self) -> str:
        return (
            f"Phase calls: {self.phase_calls} | "
            f"Budget: {self.remaining}/{self.total} remaining"
        )


class RoundManager:
    """Orchestrates the full 4-phase analysis pipeline."""

    def __init__(self, server_url: str = "http://127.0.0.1:18434"):
        self.server_url = server_url.rstrip("/")
        self.budget = CallBudget()
        self.round1_results: list[dict] = []
        self.errors: list[str] = []
        self.conflicts: list[Conflict] = []
        self.debate_outcomes: list[DebateOutcome] = []
        self.start_time: Optional[float] = None

    def _call_synthesizer(self, system: str, user: str, enable_thinking: bool = True) -> dict:
        """Make a single call to the Synthesizer model."""
        from finance_moe.config import MODELS
        import httpx

        model = MODELS["Synthesizer"].name
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 8192,
        }
        if enable_thinking:
            payload["reasoning"] = {"enabled": True}

        try:
            resp = httpx.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "text": f"ERROR: {e}"}

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            start = text.index("{") if "{" in text else 0
            end = text.rindex("}") + 1 if "}" in text else len(text)
            try:
                return json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                return {}

    def _run_phase1(self, portfolio: dict) -> None:
        """Phase 1: Run all 5 independent experts."""
        print("  Phase 1: Running 5 independent experts...", file=sys.stderr)

        from finance_moe.config import MODELS

        expert_specs = [
            ("Fee Auditor", MODELS["FeeAuditor"].name),
            ("Risk Manager", MODELS["RiskManager"].name),
            ("Allocation Strategist", MODELS["Allocator"].name),
            ("Research Analyst", MODELS["ResearchAnalyst"].name),
            ("Compliance Checker", MODELS["ComplianceChecker"].name),
        ]

        for expert_name, model_name in expert_specs:
            if not self.budget.can_afford(1):
                print(f"    ⚠ Budget exhausted before {expert_name}", file=sys.stderr)
                break

            print(f"    Running {expert_name} (model: {model_name})...", file=sys.stderr)
            try:
                # Import the expert class dynamically
                expert_classes = {
                    "Fee Auditor": "finance_moe.experts.FeeAuditor",
                    "Risk Manager": "finance_moe.experts.RiskManager",
                    "Allocation Strategist": "finance_moe.experts.Allocator",
                    "Research Analyst": "finance_moe.experts.ResearchAnalyst",
                    "Compliance Checker": "finance_moe.experts.ComplianceChecker",
                }
                module_path, class_name = expert_classes[expert_name].rsplit(".", 1)
                mod = __import__(module_path, fromlist=[class_name])
                ExpertClass = getattr(mod, class_name)

                expert = ExpertClass()
                expert.model = model_name  # Override model for this call
                output = expert.run(portfolio)
                result = output.to_dict()
                self.round1_results.append(result)
                self.budget.consume(1)
                self.budget.add_phase_call("phase1")
                print(f"      Score: {output.score}/100", file=sys.stderr)
            except Exception as e:
                error_msg = f"{expert_name} failed: {e}"
                print(f"      ERROR: {error_msg}", file=sys.stderr)
                self.errors.append(error_msg)
                self.round1_results.append({
                    "expert": expert_name,
                    "score": 0,
                    "findings": [],
                    "recommendations": [error_msg],
                    "confidence": 0.0,
                    "details": {"error": str(e)},
                })

    def _run_phase2(self, portfolio: dict) -> None:
        """Phase 2: Synthesizer detects conflicts between Phase 1 findings."""
        print("  Phase 2: Detecting conflicts...", file=sys.stderr)

        if not self.round1_results:
            print("    No Phase 1 results — skipping conflict detection.", file=sys.stderr)
            return

        # Build portfolio summary
        total = sum(acc.get("balance", 0) for acc in portfolio.get("accounts", []))
        portfolio_summary = (
            f"Total value: ${total:,.2f}\n"
            f"Risk tolerance: {portfolio.get('risk_tolerance', 'N/A')}\n"
            f"Time horizon: {portfolio.get('time_horizon_years', 'N/A')} years\n"
            f"Accounts: {len(portfolio.get('accounts', []))}"
        )

        sys_pd, user_pd = conflict_detection_prompt(
            self.round1_results, portfolio_summary
        )

        result = self._call_synthesizer(sys_pd, user_pd, enable_thinking=True)
        self.budget.consume(1)
        self.budget.add_phase_call("phase2")

        if not result["success"]:
            print(f"    Conflict detection failed: {result['text']}", file=sys.stderr)
            self.errors.append(f"Conflict detection failed: {result['text']}")
            return

        parsed = self._parse_json_response(result["text"])
        conflicts_data = parsed.get("conflicts", [])

        for c_data in conflicts_data:
            conflict = Conflict(
                topic=c_data.get("topic", "Unknown conflict"),
                expert_a=c_data.get("expert_a", "Unknown"),
                expert_b=c_data.get("expert_b", "Unknown"),
                position_a=c_data.get("position_a", ""),
                position_b=c_data.get("position_b", ""),
                severity=c_data.get("severity", "medium"),
                data_relevant=c_data.get("data_relevant", ""),
            )
            self.conflicts.append(conflict)
            print(f"    Conflict detected: {conflict.topic} [{conflict.severity}]",
                  file=sys.stderr)

        print(f"    Total conflicts: {len(self.conflicts)}", file=sys.stderr)

    def _run_phase3(self, portfolio: dict) -> None:
        """Phase 3: Run debates for each conflict."""
        print(f"  Phase 3: Running debates ({len(self.conflicts)} conflicts)...", file=sys.stderr)

        if not self.conflicts:
            print("    No conflicts to debate.", file=sys.stderr)
            return

        round_context = RoundContext(
            portfolio=portfolio,
            round1_results=self.round1_results,
            conflicts=self.conflicts,
            debate_outcomes=[],
            call_budget=self.budget,
        )

        for i, conflict in enumerate(self.conflicts):
            if not self.budget.can_afford(4):
                print(f"    ⚠ Budget exhausted before conflict {i+1}", file=sys.stderr)
                conflict.unresolved = True
                conflict.resolution = "Budget exhausted."
                self.conflicts[i] = conflict
                break

            print(f"    Debate {i+1}/{len(self.conflicts)}: {conflict.topic}",
                  file=sys.stderr)

            debate = DebateRound(
                conflict=conflict,
                server_url=self.server_url,
                context=round_context,
                budget=self.budget,
            )

            outcome = debate.run()
            self.debate_outcomes.append(outcome)
            self.budget.add_phase_call("phase3_debate")

            if outcome.resolved:
                print(f"      Resolved: {outcome.resolution}", file=sys.stderr)
            else:
                print(f"      Unresolved: {outcome.resolution}", file=sys.stderr)

    def _run_phase4(self, portfolio: dict) -> dict:
        """Phase 4: Synthesizer writes the final report."""
        print("  Phase 4: Synthesizing final report...", file=sys.stderr)

        sys_rp, user_rp = report_prompt(
            self.round1_results, self.conflicts,
            self.debate_outcomes, portfolio
        )

        result = self._call_synthesizer(sys_rp, user_rp, enable_thinking=True)
        self.budget.consume(1)
        self.budget.add_phase_call("phase4")

        if not result["success"]:
            print(f"    Report synthesis failed: {result['text']}", file=sys.stderr)
            self.errors.append(f"Report synthesis failed: {result['text']}")
            return {"error": result["text"]}

        return {"report": result["text"], "success": True}

    def execute(self, portfolio: dict) -> dict:
        """Run the full 4-phase pipeline."""
        self.start_time = time.time()
        print(f"\nStarting Finance MoE analysis (budget: {self.budget.total} calls)...",
              file=sys.stderr)

        # Phase 1
        self._run_phase1(portfolio)

        # Phase 2
        self._run_phase2(portfolio)

        # Phase 3
        self._run_phase3(portfolio)

        # Phase 4
        report_result = self._run_phase4(portfolio)

        elapsed = time.time() - self.start_time
        print(f"\nAnalysis completed in {elapsed:.1f}s", file=sys.stderr)
        print(f"Budget: {self.budget.get_summary()}", file=sys.stderr)
        print(f"Errors: {len(self.errors)}", file=sys.stderr)

        return {
            "portfolio": portfolio,
            "round1_results": self.round1_results,
            "conflicts": [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.conflicts],
            "debate_outcomes": [o.__dict__ for o in self.debate_outcomes],
            "report": report_result,
            "errors": self.errors,
            "budget": self.budget.get_summary(),
            "elapsed_seconds": elapsed,
        }

    def get_progress(self) -> str:
        """Return a status string for logging."""
        return (
            f"Phase 1: {len(self.round1_results)}/5 experts done | "
            f"Phase 2: {len(self.conflicts)} conflicts | "
            f"Phase 3: {len(self.debate_outcomes)} debates | "
            f"Budget: {self.budget.remaining}/{self.budget.total}"
        )

    def get_results(self) -> dict:
        """Return all results for formatting."""
        return {
            "round1_results": self.round1_results,
            "conflicts": self.conflicts,
            "debate_outcomes": self.debate_outcomes,
            "errors": self.errors,
            "budget": self.budget,
        }
