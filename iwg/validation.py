"""
Validation Committee — Three-Tier Quality Control.

Implements the multi-tier validation protocol described in the paper:
- Tier 1 (Solvability): open-source model verifies environment provides sufficient context
- Tier 2 (Consistency): proprietary model confirms reasoning path is reproducible
- Tier 3 (Expert Review): human-in-the-loop verifies factual correctness and logic

Only instances that pass all three tiers are included in the final benchmark.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from .models import (
    BenchmarkInstance,
    Checkpoint,
    EnvironmentInfo,
    ValidationReport,
    ValidationResult,
    ValidationTier,
    WrapperOutput,
)
from .prompts import (
    VALIDATION_TIER1_PROMPT,
    VALIDATION_TIER2_PROMPT,
    VALIDATION_TIER3_PROMPT,
)

logger = logging.getLogger(__name__)


class ValidationCommittee:
    """Three-tier validation pipeline for IWG-synthesized benchmark instances."""

    def __init__(
        self,
        tier1_model: str = "",
        tier2_model: str = "",
        tier3_callback: Optional[Callable[[str], str]] = None,
        llm_callable: Optional[Callable[[str, str], str]] = None,
    ):
        """
        Args:
            tier1_model: Open-source model name for solvability check.
            tier2_model: Proprietary model name for consistency check.
            tier3_callback: Human-in-the-loop callback (receives prompt, returns decision).
            llm_callable: Function(model_name, prompt) -> response for tiers 1&2.
        """
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.tier3_callback = tier3_callback
        self._llm = llm_callable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, instance: BenchmarkInstance) -> ValidationReport:
        """Run the full three-tier validation on a benchmark instance.

        Returns a ValidationReport. The instance is updated in-place with the report.
        """
        report_id = instance.id
        results: list[ValidationResult] = []

        # Tier 1 — Solvability
        t1 = self._run_tier1(instance)
        results.append(t1)
        if not t1.passed:
            logger.info("Instance %s failed Tier 1 (Solvability)", report_id)
            report = ValidationReport(instance_id=report_id, results=results, passed=False)
            instance.validation_report = report
            return report

        # Tier 2 — Consistency
        t2 = self._run_tier2(instance, t1)
        results.append(t2)
        if not t2.passed:
            logger.info("Instance %s failed Tier 2 (Consistency)", report_id)
            report = ValidationReport(instance_id=report_id, results=results, passed=False)
            instance.validation_report = report
            return report

        # Tier 3 — Human Expert
        t3 = self._run_tier3(instance, results)
        results.append(t3)
        passed = t3.passed

        report = ValidationReport(instance_id=report_id, results=results, passed=passed)
        instance.validation_report = report
        logger.info("Instance %s validation: %s", report_id, "PASSED" if passed else "FAILED")
        return report

    def validate_batch(self, instances: list[BenchmarkInstance]) -> list[BenchmarkInstance]:
        """Validate a batch of instances. Returns only those that passed all tiers."""
        valid: list[BenchmarkInstance] = []
        for inst in instances:
            report = self.validate(inst)
            if report.is_fully_validated:
                valid.append(inst)
        logger.info(
            "Batch validation: %d/%d instances passed all tiers",
            len(valid), len(instances),
        )
        return valid

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    def _run_tier1(self, instance: BenchmarkInstance) -> ValidationResult:
        """Tier 1: Solvability Check using open-source model."""
        prompt = self._build_tier1_prompt(instance)
        response = self._call_llm(self.tier1_model, prompt)

        parsed = self._parse_verdict(response)
        passed = parsed.get("overall_verdict", "").upper() == "PASS"

        return ValidationResult(
            tier=ValidationTier.TIER_1,
            passed=passed,
            model_name=self.tier1_model,
            reasoning=parsed.get("overall_reasoning", response[:200]),
            score=1.0 if passed else 0.0,
        )

    def _run_tier2(
        self, instance: BenchmarkInstance, tier1_result: ValidationResult
    ) -> ValidationResult:
        """Tier 2: Consistency Check using proprietary model."""
        prompt = self._build_tier2_prompt(instance, tier1_result)
        response = self._call_llm(self.tier2_model, prompt)

        parsed = self._parse_verdict(response)
        passed = parsed.get("overall_verdict", "").upper() == "PASS"

        return ValidationResult(
            tier=ValidationTier.TIER_2,
            passed=passed,
            model_name=self.tier2_model,
            reasoning=parsed.get("overall_reasoning", response[:200]),
            score=1.0 if passed else 0.0,
        )

    def _run_tier3(
        self, instance: BenchmarkInstance, prior_results: list[ValidationResult]
    ) -> ValidationResult:
        """Tier 3: Human-in-the-Loop Expert Review."""
        prompt = self._build_tier3_prompt(instance, prior_results)

        if self.tier3_callback:
            response = self.tier3_callback(prompt)
        else:
            logger.info("Tier 3 (Human Review) prompt ready. No callback configured.")
            logger.info("Tier 3 prompt:\n%s", prompt)
            response = "APPROVE"  # default pass for automated testing

        passed = "APPROVE" in response.upper()

        return ValidationResult(
            tier=ValidationTier.TIER_3,
            passed=passed,
            model_name="human-expert",
            reasoning=response[:500],
            score=1.0 if passed else 0.0,
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tier1_prompt(instance: BenchmarkInstance) -> str:
        env_desc = "\n\n".join(
            f"### Step {ei.step_index} ({ei.agent_name})\n"
            f"**Prompt**: {ei.tool_prompt}\n"
            f"**Environment Output**: {ei.tool_output}"
            for ei in instance.environments
        )
        cp_desc = "\n".join(
            f"- {cp.id}: expected='{cp.expected_value}' (type={cp.checkpoint_type.value})"
            for cp in instance.gold_checkpoints
        )
        task_desc = "\n".join(
            f"- Step {ei.step_index}: {ei.agent_name} — {ei.tool_prompt}"
            for ei in instance.environments
        )
        return VALIDATION_TIER1_PROMPT.format(
            environment_info=env_desc,
            task_steps=task_desc,
            checkpoints=cp_desc,
        )

    @staticmethod
    def _build_tier2_prompt(
        instance: BenchmarkInstance, tier1_result: ValidationResult
    ) -> str:
        env_desc = "\n\n".join(
            f"### Step {ei.step_index} ({ei.agent_name})\n"
            f"**Prompt**: {ei.tool_prompt}\n"
            f"**Environment Output**: {ei.tool_output}"
            for ei in instance.environments
        )
        cp_desc = "\n".join(
            f"- {cp.id}: expected='{cp.expected_value}'"
            for cp in instance.gold_checkpoints
        )
        task_desc = "\n".join(
            f"- Step {ei.step_index}: {ei.agent_name} — {ei.tool_prompt}"
            for ei in instance.environments
        )
        return VALIDATION_TIER2_PROMPT.format(
            environment_info=env_desc,
            task_steps=task_desc,
            checkpoints=cp_desc,
            tier1_results=f"Tier 1 passed={tier1_result.passed}, reasoning={tier1_result.reasoning}",
        )

    @staticmethod
    def _build_tier3_prompt(
        instance: BenchmarkInstance, prior_results: list[ValidationResult]
    ) -> str:
        summary_lines = []
        for r in prior_results:
            summary_lines.append(
                f"- **{r.tier.value}** ({r.model_name}): "
                f"{'PASSED' if r.passed else 'FAILED'} — {r.reasoning[:100]}"
            )
        return VALIDATION_TIER3_PROMPT.format(
            query=instance.seed_data.query,
            answer=instance.seed_data.answer,
            agent_sequence=" → ".join(instance.gold_agent_sequence),
            num_steps=len(instance.environments),
            tier1_tier2_summary="\n".join(summary_lines),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_llm(self, model: str, prompt: str) -> str:
        if self._llm:
            return self._llm(model, prompt)
        logger.warning(
            "ValidationCommittee: no LLM callable configured. "
            "Returning default PASS for model=%s.", model
        )
        return '{"overall_verdict": "PASS", "overall_reasoning": "LLM not configured; default pass."}'

    @staticmethod
    def _parse_verdict(raw: str) -> dict:
        try:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass
        # Heuristic fallback
        raw_upper = raw.upper()
        passed = "PASS" in raw_upper and "FAIL" not in raw_upper
        return {"overall_verdict": "PASS" if passed else "FAIL", "overall_reasoning": raw[:200]}
