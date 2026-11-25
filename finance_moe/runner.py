#!/usr/bin/env python3
"""Runner — entry point for the Finance MoE debate system.

Usage:
    python finance_moe/runner.py data/sample_portfolio.json --format human
    python finance_moe/runner.py data/sample_portfolio.json --format json
"""

import argparse
import json
import sys
import time

from finance_moe.rounds import RoundManager


def load_portfolio(path: str) -> dict:
    """Load portfolio data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def format_human(results: dict) -> str:
    """Format results as human-readable report."""
    lines = []

    # Executive section
    report = results.get("report", {})
    if isinstance(report, dict) and "report" in report:
        lines.append("=" * 80)
        lines.append("FINANCIAL MoE DEBATE REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(report["report"])
    else:
        lines.append("=== FINANCIAL MoE DEBATE REPORT ===")
        lines.append("")

    # Expert scores
    lines.append("")
    lines.append("-" * 60)
    lines.append("EXPERT SCORES")
    lines.append("-" * 60)
    for r in results.get("round1_results", []):
        expert = r.get("expert", "Unknown")
        score = r.get("score", 0)
        lines.append(f"  {expert}: {score}/100")

    # Conflicts
    lines.append("")
    lines.append("-" * 60)
    lines.append("CONFLICTS")
    lines.append("-" * 60)
    for c in results.get("conflicts", []):
        status = "RESOLVED" if c.get("resolved") else ("UNRESOLVED" if c.get("unresolved") else "IN PROGRESS")
        lines.append(f"  [{status}] {c.get('topic', 'Unknown')}")
        lines.append(f"    {c.get('expert_a', '')} vs {c.get('expert_b', '')}")
        lines.append(f"    Severity: {c.get('severity', 'unknown')}")
        if c.get("resolution"):
            lines.append(f"    Resolution: {c['resolution'][:200]}")

    # Budget
    lines.append("")
    lines.append("-" * 60)
    lines.append("EXECUTION METRICS")
    lines.append("-" * 60)
    lines.append(f"  {results.get('budget', 'N/A')}")
    lines.append(f"  Elapsed: {results.get('elapsed_seconds', 0):.1f}s")
    if results.get("errors"):
        lines.append(f"  Errors: {len(results['errors'])}")
        for e in results["errors"]:
            lines.append(f"    - {e}")

    # Appendix: Debate transcripts
    lines.append("")
    lines.append("=" * 80)
    lines.append("APPENDIX: DEBATE TRANSCRIPTS")
    lines.append("=" * 80)

    for outcome in results.get("debate_outcomes", []):
        # outcome is a dict (from __dict__ of DebateOutcome)
        transcript = outcome.get("transcript", [])
        if isinstance(transcript, list):
            # If transcript entries are dicts, render them
            for tr in transcript:
                if isinstance(tr, dict):
                    round_num = tr.get("round_num", "?")
                    expert_a = outcome.get("conflict", {}).get("expert_a", "?") if isinstance(outcome.get("conflict"), dict) else "?"
                    expert_b = outcome.get("conflict", {}).get("expert_b", "?") if isinstance(outcome.get("conflict"), dict) else "?"
                    lines.append("")
                    lines.append(f"CONFLICT: {outcome.get('conflict', {}).get('topic', '?') if isinstance(outcome.get('conflict'), dict) else '?'}")
                    lines.append(f"{expert_a} ↔ {expert_b}")
                    lines.append(f"Round {round_num}")
                    lines.append("-" * 40)
                    s2a = tr.get("stage2_a", {})
                    if isinstance(s2a, dict):
                        lines.append(f"  {expert_a} position: {s2a.get('position', 'N/A')}")
                    s2b = tr.get("stage2_b", {})
                    if isinstance(s2b, dict):
                        lines.append(f"  {expert_b} position: {s2b.get('position', 'N/A')}")
    else:
        lines.append("  No debate transcripts available.")

    lines.append("")
    return "\n".join(lines)


def format_json(results: dict) -> str:
    """Format results as JSON."""
    # Make everything JSON-serializable
    serializable = {}
    for k, v in results.items():
        if k == "debate_outcomes":
            serializable[k] = []
            for o in v:
                serializable[k].append(o.to_dict() if hasattr(o, 'to_dict') else o.__dict__)
        elif k == "budget":
            # Budget is a string summary
            serializable[k] = v if isinstance(v, str) else str(v)
        elif k == "report":
            if isinstance(v, dict):
                serializable[k] = v
            else:
                serializable[k] = {"report": str(v)}
        else:
            serializable[k] = v
    return json.dumps(serializable, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Finance MoE Debate System",
    )
    parser.add_argument(
        "portfolio",
        help="Path to portfolio JSON file",
    )
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:18434",
        help="LLM server URL (default: http://127.0.0.1:18434)",
    )
    args = parser.parse_args()

    # Load portfolio
    portfolio = load_portfolio(args.portfolio)

    # Run analysis
    manager = RoundManager(server_url=args.server)
    results = manager.execute(portfolio)

    # Format output
    if args.format == "json":
        output = format_json(results)
    else:
        output = format_human(results)

    print(output)


if __name__ == "__main__":
    main()
