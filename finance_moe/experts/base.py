"""Base class for LLM-powered experts.

Each expert calls its assigned LLM with a system prompt and portfolio data.
Python is available as a tool for math calculations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExpertOutput:
    """Structured output from any expert."""
    expert_name: str
    category: str
    score: float           # 0-100, higher = better
    findings: list[dict[str, Any]]
    recommendations: list[str]
    confidence: float      # 0.0-1.0
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert": self.expert_name,
            "category": self.category,
            "score": self.score,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "details": self.raw,
        }


def _python_tool(code: str) -> str:
    """Execute Python code and return stdout or error.

    This is the tool that the LLM calls when it needs to do math.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "OK (no output)"
        return f"ERROR: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out (30s limit)"
    except Exception as e:
        return f"ERROR: {e}"


class LLMPoweredExpert(ABC):
    """An expert backed by an LLM with Python tool access.

    Subclass this, set the class attributes, and implement
    `format_data()` to convert portfolio_data into the prompt text.
    """

    name: str = "base"
    category: str = "general"
    model: str = "Qwen3.8-27B-Q4_K_M.gguf"
    endpoint: str = "http://127.0.0.1:18434/v1"
    system_prompt: str = ""
    use_python_tool: bool = True

    def run(self, portfolio_data: dict) -> ExpertOutput:
        """Run the expert analysis via LLM tool-use loop.

        Returns:
            ExpertOutput parsed from LLM response.
        """
        prompt = self._build_prompt(portfolio_data)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Tool-use loop: let the LLM call Python as needed
        max_iterations = 10
        for _ in range(max_iterations):
            response = self._chat(messages)
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Check if the LLM used Python in its response
            if "<python>" in text and "</python>" in text:
                code = text[text.index("<python>") + 9: text.index("</python>")]
                tool_result = _python_tool(code)

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "tool",
                    "tool_call_id": "python_1",
                    "name": "python",
                    "content": tool_result,
                })
                continue  # let the LLM continue

            # LLM finished — no more Python calls needed
            break

        return self._parse_output(text, portfolio_data)

    def run_with_context(
        self,
        system: str,
        user_prompt: str,
        enable_thinking: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        """Run a single LLM call with custom system/user prompts.

        This is used for debate protocol stages, where we need full
        control over the prompt and thinking toggle.

        Args:
            system: System prompt string.
            user_prompt: User prompt string.
            enable_thinking: If True, enables reasoning/thinking mode.
            max_tokens: Max tokens in response.

        Returns:
            Raw LLM response text.
        """
        import httpx

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
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
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text
        except Exception as e:
            return f"ERROR: {e}"

    def _build_prompt(self, portfolio_data: dict) -> str:
        """Convert portfolio data into prompt text."""
        return json.dumps(portfolio_data, indent=2, default=str)

    def _chat(self, messages: list[dict]) -> dict:
        """Call the LLM via OpenAI-compatible API."""
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        if self.use_python_tool:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Execute Python code for calculations. Return Python code between <python> tags.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            },
                        },
                        "required": ["code"],
                    },
                },
            }]

        resp = httpx.post(
            f"{self.endpoint}/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_output(self, text: str, portfolio_data: dict) -> ExpertOutput:
        """Parse the LLM's text response into ExpertOutput.

        The LLM should return structured JSON. Fallback to defaults if parsing fails.
        """
        try:
            # Extract JSON from the LLM response (may be wrapped in markdown)
            json_str = text.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                # Remove first and last lines (``` markers)
                json_str = "\n".join(lines[1:-1])
            data = json.loads(json_str)

            return ExpertOutput(
                expert_name=self.name,
                category=self.category,
                score=float(data.get("score", 50.0)),
                findings=data.get("findings", []),
                recommendations=data.get("recommendations", []),
                confidence=float(data.get("confidence", 0.8)),
                raw={
                    "llm_response": text[:500],
                    "model": self.model,
                },
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback: LLM didn't return clean JSON
            return ExpertOutput(
                expert_name=self.name,
                category=self.category,
                score=50.0,
                findings=[{"metric": "raw_output", "value": text[:2000]}],
                recommendations=["Review output manually — LLM response was not structured JSON."],
                confidence=0.0,
                raw={"llm_response": text},
            )
