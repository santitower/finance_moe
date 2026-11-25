"""Rule-based consensus: aggregate expert scores and flag conflicts."""

from __future__ import annotations

from finance_moe.experts.base import ExpertOutput


# Category weights for overall score
_CATEGORY_WEIGHTS = {
    "fees": 0.15,
    "risk": 0.30,
    "allocation": 0.25,
    "synthesis": 0.30,
}

# Score thresholds
_THRESHOLD_HIGH = 70
_THRESHOLD_LOW = 40


def aggregate(experts: list[ExpertOutput]) -> dict:
    """Produce a consolidated view of all expert outputs."""
    # Weighted overall score
    weighted_sum = 0.0
    total_weight = 0.0
    scores_by_category = {}

    for eo in experts:
        w = _CATEGORY_WEIGHTS.get(eo.category, 0.1)
        weighted_sum += eo.score * w
        total_weight += w
        scores_by_category[eo.category] = eo.score

    overall_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0

    # Detect conflicts: experts in the same category with score gap > 30
    conflicts = _detect_conflicts(experts)

    # Build recommendation list (deduplicated)
    all_recommendations = []
    seen = set()
    for eo in experts:
        for r in eo.recommendations:
            if r not in seen:
                seen.add(r)
                all_recommendations.append((eo.expert_name, r))

    # Health label
    if overall_score >= _THRESHOLD_HIGH:
        health = "good"
    elif overall_score >= _THRESHOLD_LOW:
        health = "needs_attention"
    else:
        health = "critical"

    return {
        "overall_score": overall_score,
        "health": health,
        "scores_by_category": scores_by_category,
        "conflicts": conflicts,
        "all_recommendations": all_recommendations,
        "expert_outputs": [eo.to_dict() for eo in experts],
    }


def _detect_conflicts(experts: list[ExpertOutput]) -> list[dict]:
    """Find experts with significant score gaps within same category."""
    conflicts = []
    by_category: dict[str, list[ExpertOutput]] = {}
    for eo in experts:
        by_category.setdefault(eo.category, []).append(eo)

    for cat, outputs in by_category.items():
        if len(outputs) < 2:
            continue
        scores = [o.score for o in outputs]
        gap = max(scores) - min(scores)
        if gap > 30:
            worst = min(outputs, key=lambda o: o.score)
            best = max(outputs, key=lambda o: o.score)
            conflicts.append({
                "category": cat,
                "description": f"{best.expert_name} ({best.score}) vs {worst.expert_name} ({worst.score})",
                "gap": round(gap, 1),
                "severity": "high" if gap > 50 else "medium",
            })
    return conflicts
