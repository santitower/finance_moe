"""DebateRound — two-stage debate between two experts with Synthesizer checkpoints."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from finance_moe.conflict import Conflict
from finance_moe.context import DebateContext
from finance_moe.prompts import (
    debate_prompt_stage1,
    debate_prompt_stage2,
    checkpoint_prompt,
)

CHAT_TIMEOUT = 300  # seconds per LLM call


@dataclass
class DebateRoundResult:
    """Result of a single debate round."""
    round_num: int
    stage1_a: str
    stage2_a: dict
    stage1_b: str
    stage2_b: dict

    def to_dict(self) -> dict:
        return {
            "round_num": self.round_num,
            "stage1_a": self.stage1_a,
            "stage2_a": self.stage2_a,
            "stage1_b": self.stage1_b,
            "stage2_b": self.stage2_b,
        }


@dataclass
class DebateOutcome:
    """Complete outcome of a debate (all rounds)."""
    conflict: Conflict
    transcript: list[DebateRoundResult] = field(default_factory=list)
    resolved: bool = False
    resolution: str = ""
    rounds_completed: int = 0

    def get_transcript_text(self) -> str:
        """Full transcript formatted for Appendix."""
        lines = []
        lines.append(f"CONFLICT: {self.conflict.topic}")
        lines.append(f"Participants: {self.conflict.expert_a} ↔ {self.conflict.expert_b}")
        lines.append(f"Severity: {self.conflict.severity}")
        lines.append(f"Rounds: {self.rounds_completed}")
        lines.append("")

        for round_result in self.transcript:
            lines.append(f"Round {round_result.round_num}")
            lines.append("-" * 60)
            lines.append(f"  {self.conflict.expert_a} Stage 1:")
            lines.append(f"    {round_result.stage1_a[:800]}")
            lines.append(f"  {self.conflict.expert_a} Stage 2 (JSON):")
            lines.append(f"    {json.dumps(round_result.stage2_a, indent=6)}")
            lines.append(f"  {self.conflict.expert_b} Stage 1:")
            lines.append(f"    {round_result.stage1_b[:800]}")
            lines.append(f"  {self.conflict.expert_b} Stage 2 (JSON):")
            lines.append(f"    {json.dumps(round_result.stage2_b, indent=6)}")
            lines.append("")

        if self.resolved:
            lines.append(f"RESOLVED: {self.resolution}")
        elif self.conflict.unresolved:
            lines.append("UNRESOLVED: Budget exhausted.")
        else:
            lines.append(f"RESOLVED: {self.resolution}")

        return "\n".join(lines)


class DebateRound:
    """Runs a two-stage debate between two experts on a single conflict."""

    def __init__(
        self,
        conflict: Conflict,
        server_url: str,
        context: Any,
        budget: Any,
    ):
        self.conflict = conflict
        self.server_url = server_url.rstrip("/") + "/v1"
        self.context = context
        self.budget = budget
        self.debate_context = DebateContext()
        self.transcript: list[DebateRoundResult] = []
        self.best_stage2_a = {}
        self.best_stage2_b = {}
        self.round_num = 0
        self.outcome: Optional[DebateOutcome] = None

    def _call_llm(
        self,
        model: str,
        system: str,
        user: str,
        enable_thinking: bool = True,
        max_tokens: int = 4096,
    ) -> dict:
        """Make a single LLM API call. No retries."""
        payload: dict[str, Any] = {
            "model": model,
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
                f"{self.server_url}/chat/completions",
                json=payload,
                timeout=CHAT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "text": text}
        except httpx.HTTPError as e:
            print(f"    LLM call failed ({model}): {e}", file=sys.stderr)
            return {"success": False, "text": f"ERROR: {e}", "model": model}
        except Exception as e:
            print(f"    LLM call error ({model}): {e}", file=sys.stderr)
            return {"success": False, "text": f"ERROR: {e}", "model": model}

    def _parse_json_response(self, text: str) -> dict:
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
            # Last resort: try to find a JSON object in the text
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                return {"position": text[:200], "confidence": 0.0,
                        "key_arguments": ["Could not parse response"], "data_points": []}

    def _get_model_for_expert(self, expert_name: str) -> str:
        """Get the GGUF model name for an expert."""
        from finance_moe.config import MODELS
        for key, cfg in MODELS.items():
            if key.lower().replace(" ", "") in expert_name.lower().replace(" ", "") or \
               cfg.role in ("fast", "heavy_reasoning", "financial_reasoning", "reasoning", "narrative"):
                return cfg.name
        return "Qwen3.8-27B-Q4_K_M.gguf"

    def run(self) -> DebateOutcome:
        """Execute the full debate loop until resolved or budget exhausted."""
        expert_a = self.conflict.expert_a
        expert_b = self.conflict.expert_b
        model_a = self._get_model_for_expert(expert_a)
        model_b = self._get_model_for_expert(expert_b)

        while self.budget.remaining > 0:
            # Check budget: need 4 calls per round
            if self.budget.remaining < 4:
                # Not enough for a full round, stop
                self.conflict.unresolved = True
                self.conflict.resolution = "Budget exhausted before resolution."
                break

            self.round_num += 1
            print(f"    Debate Round {self.round_num} ({expert_a} ↔ {expert_b})", file=sys.stderr)

            # --- Stage 1A: Expert A free-text reasoning ---
            sys_a, user_a = debate_prompt_stage1(
                expert_a, expert_b, self.conflict, self.context
            )
            result_a = self._call_llm(
                model_a, sys_a, user_a, enable_thinking=True
            )
            if not result_a["success"]:
                self.conflict.unresolved = True
                self.conflict.resolution = f"{expert_a} Stage 1 call failed."
                break
            stage1_a = result_a["text"]
            self.budget.consume(1)

            # --- Stage 2A: Expert A JSON extraction ---
            stage1_a_ref = self.best_stage2_a if self.best_stage2_a else None
            sys_a2, user_a2 = debate_prompt_stage2(
                expert_a, stage1_a, stage1_a_ref
            )
            result_a2 = self._call_llm(
                model_a, sys_a2, user_a2, enable_thinking=False
            )
            if not result_a2["success"]:
                self.conflict.unresolved = True
                self.conflict.resolution = f"{expert_a} Stage 2 call failed."
                break
            stage2_a = self._parse_json_response(result_a2["text"])
            self.best_stage2_a = stage2_a
            self.budget.consume(1)

            # --- Stage 1B: Expert B free-text reasoning ---
            sys_b, user_b = debate_prompt_stage1(
                expert_b, expert_a, self.conflict, self.context
            )
            result_b = self._call_llm(
                model_b, sys_b, user_b, enable_thinking=True
            )
            if not result_b["success"]:
                self.conflict.unresolved = True
                self.conflict.resolution = f"{expert_b} Stage 1 call failed."
                break
            stage1_b = result_b["text"]
            self.budget.consume(1)

            # --- Stage 2B: Expert B JSON extraction ---
            stage1_b_ref = self.best_stage2_b if self.best_stage2_b else None
            sys_b2, user_b2 = debate_prompt_stage2(
                expert_b, stage1_b, stage1_b_ref
            )
            result_b2 = self._call_llm(
                model_b, sys_b2, user_b2, enable_thinking=False
            )
            if not result_b2["success"]:
                self.conflict.unresolved = True
                self.conflict.resolution = f"{expert_b} Stage 2 call failed."
                break
            stage2_b = self._parse_json_response(result_b2["text"])
            self.best_stage2_b = stage2_b
            self.budget.consume(1)

            # Record this round
            round_result = DebateRoundResult(
                round_num=self.round_num,
                stage1_a=stage1_a,
                stage2_a=stage2_a,
                stage1_b=stage1_b,
                stage2_b=stage2_b,
            )
            self.transcript.append(round_result)
            self.debate_context.add_round(
                self.round_num, expert_a, expert_b,
                stage1_a, stage2_a, stage1_b, stage2_b
            )

            # Update context for next round
            self.context.debate_outcomes.append({
                "expert": expert_a,
                "conflict_topic": self.conflict.topic,
                "stage2_a": stage2_a,
            })
            self.context.debate_outcomes.append({
                "expert": expert_b,
                "conflict_topic": self.conflict.topic,
                "stage2_b": stage2_b,
            })

            # --- Check for immediate consensus ---
            if self._check_consensus(stage2_a, stage2_b):
                self.conflict.resolved = True
                self.conflict.resolution = (
                    f"Both experts converged: {stage2_a.get('position', 'N/A')}"
                )
                self.conflict.rounds_played = self.round_num
                self.outcome = DebateOutcome(
                    conflict=self.conflict,
                    transcript=self.transcript,
                    resolved=True,
                    resolution=self.conflict.resolution,
                    rounds_completed=self.round_num,
                )
                print(f"    ✓ Consensus reached at round {self.round_num}", file=sys.stderr)
                return self.outcome

            # --- Synthesizer checkpoint every 4 rounds ---
            if self.round_num % 4 == 0:
                checkpoint = self._run_checkpoint(model_a, model_b, expert_a, expert_b)
                if checkpoint and checkpoint.get("recommendation") == "stop":
                    self.conflict.resolved = True
                    self.conflict.resolution = checkpoint.get("reason", "Checkpoint said stop.")
                    self.conflict.rounds_played = self.round_num
                    self.outcome = DebateOutcome(
                        conflict=self.conflict,
                        transcript=self.transcript,
                        resolved=True,
                        resolution=self.conflict.resolution,
                        rounds_completed=self.round_num,
                    )
                    print(f"    ✓ Checkpoint stop at round {self.round_num}: {self.conflict.resolution}",
                          file=sys.stderr)
                    return self.outcome

            print(f"    Round {self.round_num} complete (budget remaining: {self.budget.remaining})",
                  file=sys.stderr)

        # Budget exhausted
        self.conflict.unresolved = True
        self.conflict.resolution = (
            f"Budget exhausted after {self.round_num} rounds. "
            f"A: {self.best_stage2_a.get('position', 'N/A')}, "
            f"B: {self.best_stage2_b.get('position', 'N/A')}"
        )
        self.outcome = DebateOutcome(
            conflict=self.conflict,
            transcript=self.transcript,
            resolved=False,
            resolution=self.conflict.resolution,
            rounds_completed=self.round_num,
        )
        return self.outcome

    def _check_consensus(self, pos_a: dict, pos_b: dict) -> bool:
        """Check if both experts have converged to the same position."""
        pos_a_text = str(pos_a.get("position", "")).lower().strip()
        pos_b_text = str(pos_b.get("position", "")).lower().strip()
        # Simple consensus: positions are very similar
        if pos_a_text == pos_b_text:
            return True
        # Check if one position contains the other (convergence)
        if pos_a_text and pos_b_text:
            if len(pos_a_text) > 10 and len(pos_b_text) > 10:
                if pos_a_text in pos_b_text or pos_b_text in pos_a_text:
                    return True
        return False

    def _run_checkpoint(
        self, model_a: str, model_b: str,
        expert_a: str, expert_b: str,
    ) -> Optional[dict]:
        """Run Synthesizer checkpoint to evaluate debate progress."""
        recent = self.debate_context.get_last_rounds(4)
        if not recent:
            return None

        sys_cp, user_cp = checkpoint_prompt(expert_a, expert_b, self.conflict, recent)

        # Use the Synthesizer model for checkpoint
        from finance_moe.config import MODELS
        syn_model = MODELS["Synthesizer"].name

        result = self._call_llm(syn_model, sys_cp, user_cp, enable_thinking=True)
        if not result["success"]:
            print(f"    Checkpoint call failed: {result['text']}", file=sys.stderr)
            return None

        return self._parse_json_response(result["text"])
