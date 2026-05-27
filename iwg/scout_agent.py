"""
Scout Agent — Inverse Planning & Decomposition.

The Scout Agent serves as the architect of the reverse trajectory. It accepts
Seed Data and Target MAS Configuration, then performs capability-aware inverse
analysis to recursively deduce the necessary intermediate tasks.

Key behaviors:
1. Capability-Aware Inverse Analysis — profile agents, find logical entry points
2. Task Decomposition — break workflow into Task Marks (DAG)
3. Task Extension — proactively generate tasks to exercise all agent capabilities
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .models import (
    AgentCapability,
    MASConfig,
    ScoutPlan,
    SeedData,
    TaskMark,
)
from .prompts import SCOUT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ScoutAgent:
    """Inverse planning agent that decomposes a QA pair into a DAG of TaskMarks.

    The Scout works *backward* from the final answer, identifying which agent
    capabilities are needed at each step and constructing a verifiable workflow.

    This is a prompt-driven agent designed to be used with an LLM backend.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, seed: SeedData, mas_config: MASConfig) -> ScoutPlan:
        """Decompose seed data into a DAG of TaskMarks.

        Args:
            seed: The pre-verified QA pair.
            mas_config: The target MAS configuration (available executors).

        Returns:
            A ScoutPlan containing the DAG of TaskMarks.
        """
        logger.info("Scout Agent: planning inverse trajectory for '%s'", seed.query)

        # Build the structured prompt for the LLM
        prompt = self._build_prompt(seed, mas_config)

        # In a real implementation, this calls the LLM.
        # Here we provide both: the prompt-building logic (for integration)
        # and a rule-based fallback for offline testing / demonstration.
        raw_output = self._call_llm(prompt)
        return self._parse_output(raw_output, seed)

    def plan_rule_based(self, seed: SeedData, mas_config: MASConfig) -> ScoutPlan:
        """Rule-based fallback that decomposes without an LLM call.

        Uses the seed answer and agent capabilities to heuristically
        construct a reasonable TaskMark DAG. Useful for testing and
        for cases where the domain pattern is well-understood.
        """
        executors = mas_config.executors
        marks: list[TaskMark] = []

        # Build a capability → executor lookup
        cap_map: dict[AgentCapability, str] = {}
        for ex in executors:
            for cap in ex.capabilities:
                if cap not in cap_map:
                    cap_map[cap] = ex.name

        # Heuristic: parse the query for clues about needed steps
        query_lower = seed.query.lower()

        # Step pattern recognition
        steps = self._infer_steps(query_lower, seed.answer, cap_map, executors)

        for i, step_info in enumerate(steps):
            deps = [f"M_{j}" for j in range(i - 1, -1, -1)] if i > 0 else []
            marks.append(TaskMark(
                id=f"M_{i}",
                description=step_info["description"],
                assigned_agent=step_info["agent"],
                required_capability=step_info["capability"],
                dependencies=deps,
                checkpoint_hint=step_info.get("checkpoint_hint", ""),
                is_extension=step_info.get("is_extension", False),
            ))

        # Task extension: add a step for any unused capability
        used_caps = {m.required_capability for m in marks}
        for cap, agent_name in cap_map.items():
            if cap not in used_caps:
                ext_idx = len(marks)
                marks.append(TaskMark(
                    id=f"M_{ext_idx}",
                    description=f"[Extended] Exercise {cap.value} capability using results from previous steps",
                    assigned_agent=agent_name,
                    required_capability=cap,
                    dependencies=[f"M_{ext_idx - 1}"],
                    checkpoint_hint=f"Verification of {cap.value} operation result",
                    is_extension=True,
                ))

        return ScoutPlan(seed_data=seed, task_marks=marks)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, seed: SeedData, mas_config: MASConfig) -> str:
        executor_desc = self._format_executors(mas_config)
        user_prompt = f"""## Seed Data
- **Query**: {seed.query}
- **Answer**: {seed.answer}
- **Domain**: {seed.domain}

## Target MAS Configuration
{executor_desc}

## Instructions
Starting from the final answer "{seed.answer}", work backward to decompose
the query "{seed.query}" into a sequence of Task Marks. Each Task Mark must
be executable by exactly one of the available agents. Ensure all agent
capabilities are exercised (add extension tasks if needed)."""
        return f"{SCOUT_SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"

    @staticmethod
    def _format_executors(mas_config: MASConfig) -> str:
        lines = []
        for ex in mas_config.executors:
            caps = ", ".join(c.value for c in ex.capabilities)
            tools = ", ".join(ex.tools) if ex.tools else "none"
            lines.append(
                f"- **{ex.name}**: capabilities=[{caps}], tools=[{tools}]\n"
                f"  {ex.description}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM interface (override or inject in production)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the constructed prompt.

        Override this method or inject a callable for production use.
        The default returns a placeholder indicating the prompt was built.
        """
        logger.warning(
            "ScoutAgent._call_llm: no LLM backend configured. "
            "Use plan_rule_based() or inject an LLM callable."
        )
        return '{"reasoning": "LLM_NOT_CONFIGURED", "task_marks": []}'

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, raw: str, seed: SeedData) -> ScoutPlan:
        """Parse the LLM JSON output into a ScoutPlan."""
        try:
            data = self._extract_json(raw)
            marks = []
            for m in data.get("task_marks", []):
                cap_str = m.get("required_capability", "entity_retrieval")
                try:
                    cap = AgentCapability(cap_str)
                except ValueError:
                    cap = AgentCapability.ENTITY_RETRIEVAL

                marks.append(TaskMark(
                    id=m.get("id", f"M_{len(marks)}"),
                    description=m.get("description", ""),
                    assigned_agent=m.get("assigned_agent", ""),
                    required_capability=cap,
                    dependencies=m.get("dependencies", []),
                    checkpoint_hint=m.get("checkpoint_hint", ""),
                    is_extension=m.get("is_extension", False),
                ))
            return ScoutPlan(
                seed_data=seed,
                task_marks=marks,
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Scout output: %s", e)
            return ScoutPlan(seed_data=seed, task_marks=[])

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON object from raw LLM output (handles markdown fences)."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try to find JSON in markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try to find the outermost braces
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}

    # ------------------------------------------------------------------
    # Rule-based step inference heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_steps(
        query_lower: str,
        answer: str,
        cap_map: dict[AgentCapability, str],
        executors: list,
    ) -> list[dict]:
        """Heuristically infer required steps from the query pattern."""
        steps: list[dict] = []

        has_vision = AgentCapability.VISION in cap_map
        has_entity = AgentCapability.ENTITY_RETRIEVAL in cap_map
        has_gui = AgentCapability.GUI_OPERATION in cap_map
        has_file = AgentCapability.FILE_MANAGEMENT in cap_map
        has_data = AgentCapability.STRUCTURED_DATA in cap_map
        has_quant = AgentCapability.QUANTITATIVE in cap_map
        has_audio = AgentCapability.AUDIO_PROCESSING in cap_map
        has_search = AgentCapability.WEB_SEARCH in cap_map

        # Pattern: "what year was the director of <film> born?"
        if "director" in query_lower and ("born" in query_lower or "year" in query_lower):
            if has_vision:
                steps.append({
                    "description": "Recognize/identify the film from visual context",
                    "agent": cap_map[AgentCapability.VISION],
                    "capability": AgentCapability.VISION,
                    "checkpoint_hint": "Film title",
                })
            elif has_search:
                steps.append({
                    "description": "Search for the film to identify it",
                    "agent": cap_map[AgentCapability.WEB_SEARCH],
                    "capability": AgentCapability.WEB_SEARCH,
                    "checkpoint_hint": "Film title",
                })
            if has_entity:
                steps.append({
                    "description": "Retrieve director information for the identified film",
                    "agent": cap_map[AgentCapability.ENTITY_RETRIEVAL],
                    "capability": AgentCapability.ENTITY_RETRIEVAL,
                    "checkpoint_hint": "Director name",
                })
                steps.append({
                    "description": "Retrieve birth year of the director",
                    "agent": cap_map[AgentCapability.ENTITY_RETRIEVAL],
                    "capability": AgentCapability.ENTITY_RETRIEVAL,
                    "checkpoint_hint": "Birth year",
                })

        # Pattern: file/audio processing
        elif has_audio and ("audio" in query_lower or ".wav" in query_lower or "transcribe" in query_lower):
            steps.append({
                "description": "Transcribe/parse audio file to extract intent",
                "agent": cap_map[AgentCapability.AUDIO_PROCESSING],
                "capability": AgentCapability.AUDIO_PROCESSING,
                "checkpoint_hint": "Transcribed query text",
            })
            if has_file:
                steps.append({
                    "description": "List and identify relevant files",
                    "agent": cap_map[AgentCapability.FILE_MANAGEMENT],
                    "capability": AgentCapability.FILE_MANAGEMENT,
                    "checkpoint_hint": "File listing",
                })
            if has_data or has_quant:
                agent = cap_map.get(AgentCapability.QUANTITATIVE) or cap_map.get(AgentCapability.STRUCTURED_DATA)
                cap = AgentCapability.QUANTITATIVE if has_quant else AgentCapability.STRUCTURED_DATA
                steps.append({
                    "description": "Calculate/analyze the requested metric from data",
                    "agent": agent,
                    "capability": cap,
                    "checkpoint_hint": "Computed result",
                })

        # Pattern: general QA with entity lookup
        elif has_entity:
            steps.append({
                "description": "Retrieve information to answer the query",
                "agent": cap_map[AgentCapability.ENTITY_RETRIEVAL],
                "capability": AgentCapability.ENTITY_RETRIEVAL,
                "checkpoint_hint": "Retrieved fact",
            })

        # Generic: at minimum, use web search if available
        elif has_search:
            steps.append({
                "description": "Search for information to answer the query",
                "agent": cap_map[AgentCapability.WEB_SEARCH],
                "capability": AgentCapability.WEB_SEARCH,
                "checkpoint_hint": "Search result",
            })

        return steps
