"""Financial Synthesizer — detects conflicts, evaluates progress, writes final report."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from finance_moe.conflict import Conflict
from finance_moe.context import DebateContext


@dataclass
class SynthesizerResult:
    """Result from running the Synthesizer."""
    conflicts: list[Conflict] = None  # type: ignore
    progress: dict = None  # type: ignore
    report: str = ""

    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []
        if self.progress is None:
            self.progress = {}


class Synthesizer:
    """Financial Synthesizer — single LLM model that coordinates expert analysis."""

    model: str = "Qwen3.8-27B-Q4_K_M.gguf"
    endpoint: str = "http://127.0.0.1:18434/v1"

    def _call_llm(self, system: str, user: str,
                  enable_thinking: bool = True, max_tokens: int = 8192) -> str:
        """Make a single LLM API call. No retries."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        if enable_thinking:
            payload["reasoning"] = {"enabled": True}

        try:
            resp = httpx.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            print(f"Synthesizer LLM call failed: {e}", file=sys.stderr)
            return f"ERROR: {e}"
        except Exception as e:
            print(f"Synthesizer LLM call error: {e}", file=sys.stderr)
            return f"ERROR: {e}"

    def _parse_json(self, text: str) -> dict:
        """Try to extract JSON from an LLM response."""
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

    def detect_conflicts(self, all_findings: list[dict], portfolio: dict) -> list[Conflict]:
        """Phase 2: Analyze expert findings and return structured conflicts.

        The Synthesizer looks for substantive disagreements — not just
        different scores, but disagreements on facts, conclusions, or
        recommendations.

        Args:
            all_findings: List of ExpertOutput.to_dict() from Phase 1.
            portfolio: Full portfolio data.

        Returns:
            List of Conflict objects.
        """
        findings_text = []
        for f in all_findings:
            expert = f.get("expert", "Unknown")
            score = f.get("score", 0)
            findings_text.append(f"\n--- {expert} (score: {score}/100) ---")
            for finding in f.get("findings", []):
                findings_text.append(f"  {finding.get('metric', '')}: {finding.get('value', '')}")
            for rec in f.get("recommendations", []):
                findings_text.append(f"  → {rec}")

        total = sum(acc.get("balance", 0) for acc in portfolio.get("accounts", []))
        portfolio_summary = (
            f"Total value: ${total:,.2f}\n"
            f"Risk tolerance: {portfolio.get('risk_tolerance', 'N/A')}\n"
            f"Time horizon: {portfolio.get('time_horizon_years', 'N/A')} years\n"
            f"Accounts: {len(portfolio.get('accounts', []))}"
        )

        system = (
            "You are a Financial Synthesizer tasked with identifying substantive "
            "disagreements between expert analyses. Find genuine conflicts where "
            "experts disagree on facts, conclusions, or recommendations. Ignore "
            "areas where they just score differently on the same observation."
        )

        user = (
            f"Expert findings:\n{''.join(findings_text)}\n\n"
            f"Portfolio summary:\n{portfolio_summary}\n\n"
            f"Return JSON:\n"
            f'{{"conflicts": [\n'
            f'  {{"topic": "brief description",\n'
            f'   "expert_a": "name",\n'
            f'   "expert_b": "name",\n'
            f'   "position_a": "what expert A thinks",\n'
            f'   "position_b": "what expert B thinks",\n'
            f'   "severity": "high|medium|low",\n'
            f'   "data_relevant": "which portfolio data is at issue"}}\n'
            f"]}}"
        )

        response = self._call_llm(system, user, enable_thinking=True)
        parsed = self._parse_json(response)
        conflicts_data = parsed.get("conflicts", [])

        conflicts = []
        for c_data in conflicts_data:
            conflicts.append(Conflict(
                topic=c_data.get("topic", "Unknown conflict"),
                expert_a=c_data.get("expert_a", "Unknown"),
                expert_b=c_data.get("expert_b", "Unknown"),
                position_a=c_data.get("position_a", ""),
                position_b=c_data.get("position_b", ""),
                severity=c_data.get("severity", "medium"),
                data_relevant=c_data.get("data_relevant", ""),
            ))

        print(f"  Conflict detection returned {len(conflicts)} conflicts", file=sys.stderr)
        return conflicts

    def check_debate_progress(
        self,
        conflict: Conflict,
        round_history: list[dict],
    ) -> dict:
        """Synthesizer checkpoint: evaluate if debate is making progress.

        Called every 4 debate rounds.

        Args:
            conflict: The Conflict being debated.
            round_history: List of DebateRoundResult dicts (last 4 rounds).

        Returns:
            Dict with keys: positions_changed, making_progress, recommendation, reason
        """
        if not round_history:
            return {"recommendation": "continue", "reason": "No rounds yet."}

        round_summaries = []
        for r in round_history:
            s2a = r.get("stage2_a", {})
            s2b = r.get("stage2_b", {})
            if isinstance(s2a, dict):
                pos_a = s2a.get("position", "N/A")
            else:
                pos_a = str(s2a)
            if isinstance(s2b, dict):
                pos_b = s2b.get("position", "N/A")
            else:
                pos_b = str(s2b)
            round_summaries.append(
                f"Round {r.get('round_num', '?')}: "
                f"{conflict.expert_a} said: {pos_a[:200]} | "
                f"{conflict.expert_b} said: {pos_b[:200]}"
            )

        system = (
            "You are a debate moderator evaluating whether two experts are converging.\n"
            "Analyze whether the debate is making real progress toward resolution."
        )

        user = (
            f"Debate topic: {conflict.topic}\n"
            f"{conflict.expert_a} original position: {conflict.position_a}\n"
            f"{conflict.expert_b} original position: {conflict.position_b}\n\n"
            f"Recent rounds:\n\n" + "\n".join(round_summaries) +
            f"""\n\nEvaluate:
{{
  "positions_changed": true/false,
  "making_progress": true/false,
  "recommendation": "continue" or "stop",
  "reason": "one sentence explaining why"
}}

Consider: Has either expert changed their position from the original?
Is the debate going in circles, or is it converging toward agreement?
"""
        )

        response = self._call_llm(system, user, enable_thinking=True)
        return self._parse_json(response)

    def synthesize_report(
        self,
        findings: list[dict],
        conflicts: list[Conflict],
        debate_transcripts: list[dict],
        portfolio: dict,
    ) -> dict:
        """Phase 4: Synthesizer writes the final report.

        Args:
            findings: Phase 1 expert results.
            conflicts: Detected and debated conflicts.
            debate_transcripts: DebateRoundResult lists for each conflict.
            portfolio: Full portfolio data.

        Returns:
            Dict with 'report' key containing the full report text.
        """
        total = sum(acc.get("balance", 0) for acc in portfolio.get("accounts", []))
        portfolio_summary = json.dumps(portfolio, indent=2)

        conflict_summaries = []
        for c in conflicts:
            conflict_summaries.append(
                f"- {c.topic}: {c.expert_a} vs {c.expert_b} "
                f"[{c.severity}] → {c.resolution or 'unresolved'}"
            )

        system = (
            "You are a Financial Synthesizer producing a final report.\n"
            "Structure:\n"
            "1. Executive Summary (resolutions and final recommendations)\n"
            "2. Expert Scores\n"
            "3. Conflict Resolutions\n"
            "4. Final Recommendations (numbered, prioritized)\n"
            "5. Overall Health Score\n"
            "6. Appendix: Full Debate Transcripts"
        )

        user = (
            f"Portfolio data:\n{portfolio_summary}\n\n"
            f"Phase 1 Expert findings:\n" +
            "\n".join(
                f"[{f.get('expert', 'Unknown')}] Score: {f.get('score', 0)}/100"
                for f in findings
            ) + "\n\n"
            f"Conflicts detected and resolved:\n" +
            "\n".join(conflict_summaries) + "\n\n"
            f"Debate transcripts: {len(debate_transcripts)} debates\n\n"
            f"Produce the full report in this format:\n"
            f"1. Executive section with resolutions\n"
            f"2. Expert scores\n"
            f"3. Conflict resolutions\n"
            f"4. Final recommendations\n"
            f"5. Overall health score (0-100)\n"
            f"6. Appendix with full debate transcripts"
        )

        response = self._call_llm(system, user, enable_thinking=True, max_tokens=8192)
        return {"report": response, "success": True}
