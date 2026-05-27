"""
Wrapper Agent — Environment Synthesis & Task Encapsulation.

The Wrapper Agent converts the Scout's abstract Task Marks into a concrete,
executable interactive environment with deterministic verification checkpoints.

Key behaviors:
1. Environment Synthesis (EI) — materialize tool outputs and observations
2. Checkpoint Generation — embed exact-match, API-verify, or custom checkpoints
3. Exception Scenario Generation — inject plausible failures for robustness eval
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .models import (
    Checkpoint,
    CheckpointType,
    EnvironmentInfo,
    ScoutPlan,
    SeedData,
    TaskMark,
    WrapperOutput,
)
from .prompts import WRAPPER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class WrapperAgent:
    """Environment synthesis agent that materializes TaskMarks into executable
    interactive environments with embedded verification checkpoints.

    This is a prompt-driven agent designed to be used with an LLM backend.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, scout_plan: ScoutPlan) -> WrapperOutput:
        """Convert a ScoutPlan into a concrete interactive environment.

        Args:
            scout_plan: The DAG of TaskMarks from the Scout Agent.

        Returns:
            WrapperOutput with synthesized environments and checkpoints.
        """
        logger.info(
            "Wrapper Agent: synthesizing environment for %d task marks",
            len(scout_plan.task_marks),
        )

        prompt = self._build_prompt(scout_plan)
        raw_output = self._call_llm(prompt)
        return self._parse_output(raw_output, scout_plan)

    def synthesize_rule_based(self, scout_plan: ScoutPlan) -> WrapperOutput:
        """Rule-based environment synthesis without LLM.

        Constructs plausible environment info and checkpoints heuristically
        based on the task mark descriptions and the seed data.
        """
        seed = scout_plan.seed_data
        environments: list[EnvironmentInfo] = []
        exception_scenarios: list[dict] = []
        gold_recovery_plans: dict[str, str] = {}

        for i, mark in enumerate(scout_plan.task_marks):
            ei = self._synthesize_single(mark, seed, step_index=i,
                                         prev_ei=environments[-1] if environments else None)
            environments.append(ei)

        # Generate exception scenarios for robustness evaluation
        if len(environments) >= 2:
            mid = len(environments) // 2
            exception_scenarios.append({
                "step_index": mid,
                "exception_type": "NetworkTimeout",
                "description": f"Agent {environments[mid].agent_name} API returns 404 timeout",
            })
            gold_recovery_plans[f"NetworkTimeout_step{mid}"] = (
                f"Retry: re-call {environments[mid].agent_name} with same parameters"
            )

        return WrapperOutput(
            scout_plan=scout_plan,
            environments=environments,
            exception_scenarios=exception_scenarios,
            gold_recovery_plans=gold_recovery_plans,
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, scout_plan: ScoutPlan) -> str:
        marks_json = []
        for m in scout_plan.task_marks:
            marks_json.append({
                "id": m.id,
                "description": m.description,
                "assigned_agent": m.assigned_agent,
                "required_capability": m.required_capability.value,
                "dependencies": m.dependencies,
                "checkpoint_hint": m.checkpoint_hint,
                "is_extension": m.is_extension,
            })

        user_prompt = f"""## Scout Plan
```json
{json.dumps(marks_json, indent=2)}
```

## Seed Data
- **Query**: {scout_plan.seed_data.query}
- **Answer**: {scout_plan.seed_data.answer}

## Instructions
For each Task Mark above, synthesize:
1. The environmental information (tool_prompt + tool_output) the agent would see.
2. A deterministic checkpoint for verification.
3. Any plausible exception scenarios for robustness testing.

Remember: provide *evidence* from which checkpoint values can be inferred,
not the answers directly."""
        return f"{WRAPPER_SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        logger.warning(
            "WrapperAgent._call_llm: no LLM backend configured. "
            "Use synthesize_rule_based() or inject an LLM callable."
        )
        return '{"environments": [], "exception_scenarios": [], "gold_recovery_plans": {}}'

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, raw: str, scout_plan: ScoutPlan) -> WrapperOutput:
        try:
            data = ScoutAgent._extract_json(raw)  # reuse extraction logic
            environments = []
            for i, env_data in enumerate(data.get("environments", [])):
                cp_data = env_data.get("checkpoint", {})
                checkpoint = None
                if cp_data:
                    cp_type_str = cp_data.get("checkpoint_type", "exact_match")
                    try:
                        cp_type = CheckpointType(cp_type_str)
                    except ValueError:
                        cp_type = CheckpointType.EXACT_MATCH
                    checkpoint = Checkpoint(
                        task_mark_id=env_data.get("task_mark_id", ""),
                        checkpoint_type=cp_type,
                        expected_value=cp_data.get("expected_value", ""),
                        verification_prompt=cp_data.get("verification_prompt", ""),
                        step_index=i,
                    )
                environments.append(EnvironmentInfo(
                    task_mark_id=env_data.get("task_mark_id", f"M_{i}"),
                    step_index=env_data.get("step_index", i),
                    agent_name=env_data.get("agent_name", ""),
                    tool_prompt=env_data.get("tool_prompt", ""),
                    tool_output=env_data.get("tool_output", ""),
                    checkpoint=checkpoint,
                ))

            return WrapperOutput(
                scout_plan=scout_plan,
                environments=environments,
                exception_scenarios=data.get("exception_scenarios", []),
                gold_recovery_plans=data.get("gold_recovery_plans", {}),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Wrapper output: %s", e)
            return WrapperOutput(scout_plan=scout_plan, environments=[])

    # ------------------------------------------------------------------
    # Rule-based single environment synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _synthesize_single(
        mark: TaskMark,
        seed: SeedData,
        step_index: int = 0,
        prev_ei: Optional[EnvironmentInfo] = None,
    ) -> EnvironmentInfo:
        """Heuristically synthesize one EnvironmentInfo from a TaskMark."""
        cap = mark.required_capability
        agent = mark.assigned_agent

        # Build context from previous step
        prev_context = ""
        if prev_ei and prev_ei.checkpoint:
            prev_context = f"Given the previous finding that the answer involves '{prev_ei.checkpoint.expected_value}', "

        tool_prompt, tool_output, checkpoint = "", "", None

        if cap.value == "vision":
            tool_prompt = f"Observe the visual content and identify key entities related to: {seed.query}"
            tool_output = (
                f"The image shows visual content relevant to the query. "
                f"Key identifiable elements include textual and visual features "
                f"that can be recognized by a vision model."
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value=mark.checkpoint_hint or "identified entity",
                verification_prompt=f"Extract the key entity from the vision output",
                step_index=step_index,
            )

        elif cap.value == "entity_retrieval":
            # The checkpoint_hint tells us what entity we're looking up
            entity_hint = mark.checkpoint_hint or "retrieved fact"
            lookup_target = ""
            if prev_ei and prev_ei.checkpoint:
                lookup_target = prev_ei.checkpoint.expected_value

            tool_prompt = f"{prev_context}Retrieve detailed information about {lookup_target}."
            tool_output = (
                f"Entity retrieval results for '{lookup_target}':\n"
                f"The database contains structured records including attributes "
                f"such as names, dates, relationships, and metadata.\n"
                f"Relevant records have been returned for further processing."
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value=entity_hint,
                verification_prompt=f"What {entity_hint} was retrieved?",
                step_index=step_index,
            )

        elif cap.value == "gui_operation":
            tool_prompt = f"Perform the GUI operation to complete the task based on previous findings."
            tool_output = (
                f"GUI operation executed successfully.\n"
                f"Status: 200 OK\n"
                f"The requested operation has been applied to the target system."
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.API_VERIFY,
                expected_value=f"API(GET /verify) -> status=200",
                verification_prompt=f"Verify via API that the GUI operation completed successfully",
                step_index=step_index,
            )

        elif cap.value == "file_management":
            tool_prompt = f"List and access files relevant to the query."
            tool_output = (
                f"Directory listing:\n"
                f"- data.csv (12 KB, modified 2024-03-15)\n"
                f"- report.pdf (245 KB, modified 2024-03-14)\n"
                f"- notes.txt (3 KB, modified 2024-03-13)"
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value="data.csv, report.pdf",
                verification_prompt="List the files found in the directory",
                step_index=step_index,
            )

        elif cap.value == "structured_data":
            tool_prompt = f"Parse and analyze the structured data file."
            tool_output = (
                f"Data analysis results:\n"
                f"Columns: region, value, percentage, date\n"
                f"Row count: 150\n"
                f"Summary statistics computed for all numeric columns."
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value=mark.checkpoint_hint or "data parsed",
                verification_prompt="Confirm the data was successfully parsed",
                step_index=step_index,
            )

        elif cap.value == "quantitative":
            tool_prompt = f"Perform quantitative analysis based on the structured data."
            tool_output = (
                f"Calculation complete.\n"
                f"The quantitative analysis has been performed using the provided data.\n"
                f"Results include computed ratios, percentages, and statistical measures."
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value=mark.checkpoint_hint or "computed result",
                verification_prompt="What is the computed quantitative result?",
                step_index=step_index,
            )

        elif cap.value == "audio_processing":
            tool_prompt = f"Transcribe and parse the audio file to extract the user's intent."
            tool_output = (
                f"Audio transcription complete.\n"
                f"The audio contains a spoken request with specific parameters.\n"
                f"Key entities and intent have been extracted from the transcription."
            )
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value=mark.checkpoint_hint or "transcribed query",
                verification_prompt="What query was transcribed from the audio?",
                step_index=step_index,
            )

        else:
            tool_prompt = f"Execute the task: {mark.description}"
            tool_output = f"Task executed. Results available for the next step."
            checkpoint = Checkpoint(
                task_mark_id=mark.id,
                checkpoint_type=CheckpointType.EXACT_MATCH,
                expected_value=mark.checkpoint_hint or "task completed",
                verification_prompt=f"Verify completion of: {mark.description}",
                step_index=step_index,
            )

        return EnvironmentInfo(
            task_mark_id=mark.id,
            step_index=step_index,
            agent_name=agent,
            tool_prompt=tool_prompt,
            tool_output=tool_output,
            checkpoint=checkpoint,
        )


# Import needed for _parse_output
from .scout_agent import ScoutAgent
