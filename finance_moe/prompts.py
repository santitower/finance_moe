"""Two-stage debate prompt templates."""

from __future__ import annotations

import json
from typing import Any


def debate_prompt_stage1(expert_a: str, expert_b: str, conflict: Any,
                          context: Any) -> tuple[str, str]:
    """Build Stage 1 prompt for Expert A (free-text reasoning, thinking enabled).

    Returns:
        (system_prompt, user_prompt)
    """
    # Find which of the two was the original analyst for this conflict area
    # Use the RoundContext to find the original finding
    original_finding = ""
    if context and hasattr(context, "round1_results"):
        for r in context.round1_results:
            if r.get("expert") == expert_a:
                findings_text = []
                for f in r.get("findings", []):
                    findings_text.append(f"{f.get('metric', '')}: {f.get('value', '')}")
                original_finding = "; ".join(findings_text[:3])

    other_latest = ""
    if context and hasattr(context, "debate_outcomes"):
        for outcome in context.debate_outcomes:
            if outcome.get("expert") == expert_b and outcome.get("conflict_topic") == conflict.topic:
                s2b = outcome.get("stage2_b", {})
                if isinstance(s2b, dict):
                    other_latest = s2b.get("position", conflict.position_b)
                else:
                    other_latest = str(s2b)

    relevant_data = context.get_conflict_context(conflict.topic) if context else ""

    system = (
        f"You are {expert_a}. You previously analyzed the portfolio independently.\n"
        f"Now you are in a structured debate with {expert_b}.\n"
        f"This is Stage 1: Think through your reasoning in free text.\n"
        f"Challenge {expert_b}'s reasoning where you disagree.\n"
        f"Cite specific portfolio data.\n"
        f"Do NOT use JSON — write in natural language."
    )

    user = f"""CONFLICT: {conflict.topic}

You are debating with {expert_b} on this topic.

Your original analysis: "{original_finding}"
{expert_b}'s original analysis: "{conflict.position_b}"

{expert_b}'s latest position on this conflict: "{other_latest}"

Severity: {conflict.severity}

Relevant portfolio data for this conflict:
{relevant_data}

Think through your response carefully. Challenge their reasoning where you disagree.
Cite specific portfolio data. Explain if their position changed your thinking.
Respond in free text — no JSON format required.
"""
    return system, user


def debate_prompt_stage2(expert: str, stage1_response: str,
                          stage2_position: dict) -> tuple[str, str]:
    """Build Stage 2 prompt for an expert (JSON extraction, thinking disabled).

    Args:
        expert: Expert name.
        stage1_response: The free-text reasoning from Stage 1.
        stage2_position: Their current position dict (from previous round, if any).

    Returns:
        (system_prompt, user_prompt)
    """
    system = (
        f"You are {expert}.\n"
        f"Extract your position as strictly valid JSON. No markdown, no extra text."
    )

    position_ref = ""
    if stage2_position:
        position_ref = f"\nYour current best position: {stage2_position}"

    user = (
        f"Based on the reasoning above, extract your position as JSON:\n"
        f"{{"
        f'"position": "one-line stance",'
        f'"confidence": 0.0-1.0,'
        f'"key_arguments": ["short argument", "short argument"],'
        f'"data_points": ["number or metric"]'
        f"}}\n"
        f"Reasoning summary:\n{stage1_response[:2000]}"
        f"{position_ref}"
    )
    return system, user


def checkpoint_prompt(expert_a: str, expert_b: str, conflict: Any,
                       recent_rounds: list[dict]) -> tuple[str, str]:
    """Build checkpoint prompt for the Synthesizer to evaluate debate progress.

    Returns:
        (system_prompt, user_prompt)
    """
    round_summaries = []
    for r in recent_rounds:
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
            f"{expert_a} said: {pos_a[:200]} | "
            f"{expert_b} said: {pos_b[:200]}"
        )

    system = (
        "You are a debate moderator evaluating whether two experts are converging.\n"
        "Analyze whether the debate is making real progress toward resolution."
    )

    user = (
        f"Debate topic: {conflict.topic}\n"
        f"{expert_a} original position: {conflict.position_a}\n"
        f"{expert_b} original position: {conflict.position_b}\n\n"
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
    return system, user


def conflict_detection_prompt(all_findings: list[dict],
                               portfolio_summary: str) -> tuple[str, str]:
    """Build the Phase 2 prompt for the Synthesizer to detect conflicts.

    Returns:
        (system_prompt, user_prompt)
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

    system = (
        "You are a Financial Synthesizer tasked with identifying substantive "
        "disagreements between expert analyses.\n"
        "Find genuine conflicts where experts disagree on facts, conclusions, "
        "or recommendations. Ignore areas where they just score differently "
        "on the same observation."
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
    return system, user


def report_prompt(findings: list[dict], conflicts: list[Any],
                   debate_transcripts: list[dict],
                   portfolio: dict) -> tuple[str, str]:
    """Build the Phase 4 prompt for the Synthesizer to write the final report.

    Returns:
        (system_prompt, user_prompt)
    """
    portfolio_summary = json.dumps(portfolio, indent=2) if isinstance(portfolio, dict) else str(portfolio)

    conflict_summaries = []
    for c in conflicts:
        if isinstance(c, dict):
            conflict_summaries.append(
                f"- {c.get('topic', 'Unknown')}: "
                f"{c.get('expert_a', '')} vs {c.get('expert_b', '')} "
                f"[{c.get('severity', 'unknown')}] → {c.get('resolution', 'unresolved')}"
            )
        else:
            conflict_summaries.append(
                f"- {c.topic}: {c.expert_a} vs {c.expert_b} [{c.severity}] → {c.resolution or 'unresolved'}"
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
    return system, user
