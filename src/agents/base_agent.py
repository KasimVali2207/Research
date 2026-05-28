"""
Base agent class providing shared LLM interaction, prompt formatting,
and structured-output parsing for all clinical agents.

Design rationale: All agents share rate-limited LLM calls, retry logic,
and token tracking — centralizing here avoids duplication and makes
cost accounting trivial.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Fallback "LLM" used when OPENAI_API_KEY is not set
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Returns a minimal valid JSON mock so agents can run without an API key."""

    def invoke(self, messages):
        class _Resp:
            content = json.dumps({
                "abnormal_patterns": [],
                "key_trajectories": [],
                "trend_summary": "No API key — mock response.",
                "reasoning": "No API key — mock response.",
                "risk_scores": {"colorectal": {"probability": 0.3, "confidence": "low"},
                                "lung": {"probability": 0.2, "confidence": "low"},
                                "liver": {"probability": 0.15, "confidence": "low"}},
                "primary_concern": "colorectal",
                "differentials": [],
                "citations": [],
                "grounding_score": 0.5,
                "urgency": "routine",
                "recommendation": "Mock output — no OpenAI API key configured.",
            })
            usage_metadata = None
        return _Resp()


# ---------------------------------------------------------------------------
# Normal reference ranges used for ↑/↓ flagging across agents.
# ---------------------------------------------------------------------------
_DEFAULT_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "hemoglobin": (12.0, 17.5),       # g/dL — wide band covers M/F
    "hematocrit": (36.0, 52.0),       # %
    "mcv": (80.0, 100.0),             # fL
    "wbc": (4.0, 11.0),               # 10³/µL
    "neutrophils": (1.8, 7.7),        # 10³/µL (absolute)
    "lymphocytes": (1.0, 4.8),        # 10³/µL (absolute)
    "monocytes": (0.2, 1.0),
    "platelets": (150.0, 400.0),      # 10³/µL
    "alt": (7.0, 56.0),               # U/L
    "ast": (10.0, 40.0),              # U/L
    "alp": (44.0, 147.0),             # U/L
    "bilirubin_total": (0.2, 1.2),    # mg/dL
    "albumin": (3.5, 5.0),            # g/dL
    "creatinine": (0.6, 1.2),         # mg/dL
    "sodium": (136.0, 145.0),         # mEq/L
    "potassium": (3.5, 5.0),          # mEq/L
    "glucose": (70.0, 100.0),         # mg/dL fasting
    "crp": (0.0, 10.0),               # mg/L — high-sensitivity cutoff varies
    "esr": (0.0, 20.0),               # mm/hr (male) — conservative
    "ferritin": (15.0, 300.0),        # ng/mL
    "nlr": (0.0, 3.0),               # neutrophil-to-lymphocyte ratio
    "plr": (0.0, 150.0),             # platelet-to-lymphocyte ratio
    "sii": (0.0, 500.0),             # systemic immune-inflammation index
}

# Map feature names to human-readable categories for prompt formatting.
_LAB_CATEGORIES: dict[str, list[str]] = {
    "CBC": [
        "hemoglobin", "hematocrit", "mcv", "wbc",
        "neutrophils", "lymphocytes", "monocytes", "platelets",
    ],
    "Metabolic Panel": [
        "creatinine", "sodium", "potassium", "glucose", "albumin",
    ],
    "Liver Function": [
        "alt", "ast", "alp", "bilirubin_total",
    ],
    "Inflammatory Markers": [
        "crp", "esr", "ferritin", "nlr", "plr", "sii",
    ],
}


class BaseAgent(ABC):
    """Abstract base for all clinical LLM agents.

    Subclasses implement ``run`` and rely on shared helpers:
    * ``_call_llm``  — structured JSON output with retry
    * ``_format_lab_values`` — normalised lab block for prompts
    * ``log_run`` — structured timing/token log for evaluation
    """

    def __init__(
        self,
        agent_name: str,
        llm_model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> None:
        self.agent_name = agent_name
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Instantiate LLM — fall back to mock when no API key is configured
        _has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        if _has_key:
            try:
                self.llm = ChatOpenAI(
                    model=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_kwargs={"seed": 42} if temperature == 0.0 else {},
                )
            except Exception:
                self.llm = _FakeLLM()
        else:
            self.llm = _FakeLLM()
            self.logger.warning(
                "OPENAI_API_KEY not set — agent '%s' running in mock mode.", agent_name
            )

        self.logger = logging.getLogger(f"agents.{agent_name}")

        # Accumulated token counters for cost tracking.
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

        # Per-run timing list (populated by log_run).
        self._run_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, inputs: dict) -> dict:
        """Execute agent logic and return a JSON-serialisable result dict."""

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict | None = None,
    ) -> dict:
        """Call the LLM and parse a structured JSON response.

        Retries up to 3 times on JSON parse failure — GPT-4o occasionally
        wraps output in markdown fences even when instructed not to.

        Args:
            system_prompt: Role/instruction context.
            user_prompt:   Patient-specific content.
            output_schema: Optional JSON Schema dict. When provided, the
                           schema is injected into the system prompt so the
                           model knows the expected shape.

        Returns:
            Parsed dict from LLM response.

        Raises:
            RuntimeError: If all 3 retries are exhausted.
        """
        if output_schema is not None:
            schema_str = json.dumps(output_schema, indent=2)
            system_prompt = (
                f"{system_prompt}\n\n"
                f"You MUST respond ONLY with valid JSON matching this schema:\n"
                f"```json\n{schema_str}\n```\n"
                f"No prose, no markdown outside the JSON block."
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)

                # Track token usage when metadata is available.
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    self._total_prompt_tokens += usage.get("input_tokens", 0)
                    self._total_completion_tokens += usage.get("output_tokens", 0)

                raw = response.content.strip()

                # Strip markdown fences the model may emit despite instructions.
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    # Drop opening fence (may include language tag) and closing fence.
                    inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                    raw = "\n".join(inner).strip()

                parsed: dict = json.loads(raw)
                return parsed

            except json.JSONDecodeError as exc:
                last_error = exc
                self.logger.warning(
                    "JSON parse failure on attempt %d/%d for agent '%s': %s",
                    attempt + 1,
                    max_retries,
                    self.agent_name,
                    exc,
                )
                # Brief backoff before retry — avoids hammering the API.
                time.sleep(1.5 ** attempt)

            except Exception as exc:  # noqa: BLE001
                # Non-parse errors (network, quota) should surface immediately.
                self.logger.error("LLM call failed: %s", exc)
                raise

        raise RuntimeError(
            f"Agent '{self.agent_name}': JSON parse failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _format_lab_values(self, lab_dict: dict) -> str:
        """Format a flat {feature: value} dict into a structured lab block.

        Values are grouped into clinical categories and annotated with ↑/↓
        when outside the reference range, so the LLM doesn't have to do
        unit arithmetic.

        Args:
            lab_dict: ``{feature_name: numeric_value}``

        Returns:
            Multi-line string suitable for embedding in an LLM prompt.
        """
        lines: list[str] = []
        normal = _DEFAULT_NORMAL_RANGES

        categorised: set[str] = set()

        for category, features in _LAB_CATEGORIES.items():
            category_lines: list[str] = []
            for feat in features:
                if feat not in lab_dict:
                    continue
                val = lab_dict[feat]
                categorised.add(feat)

                if val is None:
                    marker = "  "
                    display = "N/A"
                elif feat in normal:
                    lo, hi = normal[feat]
                    if val > hi:
                        marker = "↑"
                    elif val < lo:
                        marker = "↓"
                    else:
                        marker = "  "
                    display = f"{val:.2f}"
                else:
                    marker = "  "
                    display = f"{val:.2f}"

                category_lines.append(f"  {marker} {feat:<25} {display}")

            if category_lines:
                lines.append(f"[{category}]")
                lines.extend(category_lines)
                lines.append("")

        # Catch-all: features not in any predefined category
        uncategorised = {k: v for k, v in lab_dict.items() if k not in categorised}
        if uncategorised:
            lines.append("[Other]")
            for feat, val in uncategorised.items():
                display = f"{val:.2f}" if isinstance(val, float) else str(val)
                lines.append(f"     {feat:<25} {display}")
            lines.append("")

        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Run logging
    # ------------------------------------------------------------------

    def log_run(self, inputs: dict, outputs: dict, duration: float) -> None:
        """Append a structured run record for offline evaluation.

        Args:
            inputs:   Agent input dict (subject_id at minimum).
            outputs:  Agent output dict.
            duration: Wall-clock seconds the agent took.
        """
        record = {
            "agent": self.agent_name,
            "subject_id": inputs.get("subject_id", "unknown"),
            "duration_sec": round(duration, 3),
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "output_keys": list(outputs.keys()),
            "timestamp": time.time(),
        }
        self._run_log.append(record)
        self.logger.debug(
            "agent=%s subject=%s duration=%.2fs prompt_tok=%d completion_tok=%d",
            self.agent_name,
            record["subject_id"],
            duration,
            self._total_prompt_tokens,
            self._total_completion_tokens,
        )

    # ------------------------------------------------------------------
    # Token / cost summary
    # ------------------------------------------------------------------

    def get_token_usage(self) -> dict[str, int]:
        """Return cumulative token counts since agent instantiation."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }
