"""Conflict dataclass — structured disagreement between two experts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Conflict:
    """A substantive disagreement between two experts on the portfolio."""
    topic: str                           # e.g. "Tech concentration level"
    expert_a: str                        # e.g. "Risk Manager"
    expert_b: str                        # e.g. "Allocator"
    position_a: str                      # What Expert A thinks
    position_b: str                      # What Expert B thinks
    severity: str                        # "high" | "medium" | "low"
    data_relevant: str = ""              # Which portfolio data is at issue
    resolution: str = ""                 # Filled in after debate
    resolved: bool = False               # Whether debate has converged
    rounds_played: int = 0               # Number of debate rounds completed
    round_history: list[dict] = field(default_factory=list)
    unresolved: bool = False             # True if budget exhausted without resolution

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "expert_a": self.expert_a,
            "expert_b": self.expert_b,
            "position_a": self.position_a,
            "position_b": self.position_b,
            "severity": self.severity,
            "data_relevant": self.data_relevant,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "rounds_played": self.rounds_played,
            "unresolved": self.unresolved,
        }
