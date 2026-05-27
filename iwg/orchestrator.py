"""
Orchestrator MAS — Plan & Reflexion Loop.

Implements the centralized Orchestrator agent with two operational modes:

- **Plan Mode**: High-level task scheduling. Analyzes global context, maintains
  a DAG task board (ADD/MODIFY operations), and dispatches parallel next_actions
  to executor agents.

- **Reflexion Mode**: Result auditing and error classification. Evaluates executor
  outputs, updates task board state, and classifies failures (Hallucination,
  MissingInfo, WrongFormat, ToolError, Timeout).

The Orchestrator produces scheduling vectors p_k at each step, which are used
to compute the scheduling entropy H(t) for the mean-field dynamics model.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Callable, Optional

from .models import (
    ExecutionStep,
    ExecutionTrajectory,
    MASConfig,
    NextAction,
    PlanOutput,
    ReflexionOutput,
    ReflexionStatus,
    TaskBoardItem,
    TaskStatus,
)
from .prompts import ORCHESTRATOR_PLAN_PROMPT, ORCHESTRATOR_REFLEXION_PROMPT

logger = logging.getLogger(__name__)


class Orchestrator:
    """Centralized Orchestrator agent for LLM-based Multi-Agent Systems.

    Operates in a Plan-Reflexion loop: at each step the Orchestrator produces a
    scheduling vector p_k over executor agents, selects one or more agents to act,
    then reflects on their outputs before proceeding to the next step.
    """

    def __init__(
        self,
        mas_config: MASConfig,
        model_name: str = "",
        llm_callable: Optional[Callable[[str], str]] = None,
    ):
        self.mas_config = mas_config
        self.model_name = model_name
        self._llm = llm_callable

        # Build executor name → index mapping for scheduling vectors
        self._executor_index: dict[str, int] = {}
        for i, ex in enumerate(mas_config.executors):
            self._executor_index[ex.name] = i

        self.n_executors = len(mas_config.executors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_query: str,
        executor_callable: Callable[[str, str], str],
        max_steps: int = 20,
        checkpoint_validator: Optional[Callable[[str, str], bool]] = None,
    ) -> ExecutionTrajectory:
        """Execute the full Plan-Reflexion loop on a user query.

        Args:
            user_query: The initial user request.
            executor_callable: Function(agent_name, input_prompt) -> output_string.
            max_steps: Maximum number of orchestration steps.
            checkpoint_validator: Optional function(expected, actual) -> bool.

        Returns:
            Complete ExecutionTrajectory with per-step scheduling vectors.
        """
        trajectory = ExecutionTrajectory(
            instance_id="",
            model_name=self.model_name,
        )

        # State initialization
        global_context: list[str] = [f"User Query: {user_query}"]
        task_board: list[TaskBoardItem] = []
        step = 0

        while step < max_steps:
            # --- Plan Phase ---
            plan = self.plan(user_query, global_context, task_board, step, max_steps)

            if not plan.next_actions:
                # No more actions — task complete
                break

            # Compute scheduling entropy for this step
            p_k = self._compute_scheduling_vector(plan)
            entropy = self._compute_entropy(p_k)

            # --- Execution Phase ---
            exec_step = ExecutionStep(
                step_index=step,
                plan_output=plan,
                scheduling_entropy=entropy,
                scheduling_vector=p_k,
            )

            # Execute all parallel actions
            for action in plan.next_actions:
                try:
                    output = executor_callable(action.agent, action.input)
                except Exception as e:
                    output = f"ERROR: {e}"

                exec_step.executor_name = action.agent
                exec_step.executor_output = output

                # --- Reflexion Phase ---
                reflexion = self.reflexion(
                    user_query, action, output, "", global_context, task_board
                )
                exec_step.reflexion_output = reflexion

                # Update task board
                self._update_task_board(task_board, action.task_id, reflexion)

            # Update context
            global_context.append(
                f"Step {step}: Agent={exec_step.executor_name}, "
                f"Output={exec_step.executor_output[:200]}"
            )
            trajectory.steps.append(exec_step)
            step += 1

        trajectory.completed = step < max_steps
        return trajectory

    # ------------------------------------------------------------------
    # Plan Mode
    # ------------------------------------------------------------------

    def plan(
        self,
        user_query: str,
        global_context: list[str],
        task_board: list[TaskBoardItem],
        step_index: int,
        max_steps: int,
    ) -> PlanOutput:
        """Generate the next scheduling decision.

        Analyzes the current global context and task board to produce:
        - A thought process explaining the decision
        - DAG task board updates (ADD/MODIFY)
        - Parallel next_actions assigned to specific executor agents
        """
        prompt = self._build_plan_prompt(
            user_query, global_context, task_board, step_index, max_steps
        )

        raw = self._call_llm(prompt)
        return self._parse_plan_output(raw)

    # ------------------------------------------------------------------
    # Reflexion Mode
    # ------------------------------------------------------------------

    def reflexion(
        self,
        user_query: str,
        last_action: NextAction,
        executor_output: str,
        expected_checkpoint: str,
        global_context: list[str],
        task_board: list[TaskBoardItem],
    ) -> ReflexionOutput:
        """Evaluate executor output and update system state.

        Classifies results as SUCCESS or FAILURE. On failure, identifies
        the error type: Hallucination, MissingInfo, WrongFormat, ToolError, Timeout.
        """
        prompt = self._build_reflexion_prompt(
            user_query, last_action, executor_output,
            expected_checkpoint, global_context, task_board,
        )
        raw = self._call_llm(prompt)
        return self._parse_reflexion_output(raw)

    # ------------------------------------------------------------------
    # Scheduling vector & entropy
    # ------------------------------------------------------------------

    def _compute_scheduling_vector(self, plan: PlanOutput) -> list[float]:
        """Compute the scheduling probability vector p_k from plan output.

        The vector represents the Orchestrator's probability distribution
        over executor agents. When multiple agents are dispatched in parallel,
        probability mass is distributed equally among them.
        """
        p = [0.0] * max(1, self.n_executors)
        if not plan.next_actions:
            return p

        n_actions = len(plan.next_actions)
        for action in plan.next_actions:
            idx = self._executor_index.get(action.agent, -1)
            if idx >= 0:
                p[idx] = 1.0 / n_actions

        return p

    @staticmethod
    def _compute_entropy(p: list[float]) -> float:
        """Shannon entropy of the scheduling distribution."""
        entropy = 0.0
        for prob in p:
            if prob > 1e-10:
                entropy -= prob * math.log(prob)
        return entropy

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_plan_prompt(
        self,
        user_query: str,
        global_context: list[str],
        task_board: list[TaskBoardItem],
        step_index: int,
        max_steps: int,
    ) -> str:
        executor_desc = "\n".join(
            f"- **{ex.name}**: {ex.description} "
            f"(capabilities: {[c.value for c in ex.capabilities]}, "
            f"tools: {ex.tools})"
            for ex in self.mas_config.executors
        )

        context_str = "\n".join(f"[{i}]: {c}" for i, c in enumerate(global_context))

        board_str = "\n".join(
            f"- {t.id}: status={t.status.value}, deps={t.dependencies}, "
            f"summary={t.result_summary}"
            for t in task_board
        ) if task_board else "(empty)"

        return ORCHESTRATOR_PLAN_PROMPT.format(
            executor_descriptions=executor_desc,
            user_query=user_query,
            step_index=step_index,
            global_context=context_str,
            task_board=board_str,
            max_steps=max_steps,
        )

    def _build_reflexion_prompt(
        self,
        user_query: str,
        last_action: NextAction,
        executor_output: str,
        expected_checkpoint: str,
        global_context: list[str],
        task_board: list[TaskBoardItem],
    ) -> str:
        executor_desc = "\n".join(
            f"- **{ex.name}**: {ex.description}"
            for ex in self.mas_config.executors
        )
        board_str = "\n".join(
            f"- {t.id}: status={t.status.value}, deps={t.dependencies}"
            for t in task_board
        ) if task_board else "(empty)"

        return ORCHESTRATOR_REFLEXION_PROMPT.format(
            executor_descriptions=executor_desc,
            user_query=user_query,
            last_action=f"Agent={last_action.agent}, Input={last_action.input}",
            executor_output=executor_output[:500],
            expected_checkpoint=expected_checkpoint or "(none)",
            task_board=board_str,
        )

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if self._llm:
            return self._llm(prompt)
        logger.warning(
            "Orchestrator._call_llm: no LLM callable configured. "
            "Returning empty plan."
        )
        return '{"thought_process": "LLM not configured", "task_board_updates": [], "next_actions": []}'

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan_output(raw: str) -> PlanOutput:
        try:
            data = Orchestrator._extract_json(raw)
            updates = []
            for u in data.get("task_board_updates", []):
                updates.append(TaskBoardItem(
                    id=u.get("id", ""),
                    action=u.get("action", "ADD"),
                    dependencies=u.get("dependencies", []),
                ))
            actions = []
            for a in data.get("next_actions", []):
                actions.append(NextAction(
                    agent=a.get("agent", ""),
                    input=a.get("input", ""),
                    task_id=a.get("task_id", ""),
                ))
            return PlanOutput(
                thought_process=data.get("thought_process", ""),
                task_board_updates=updates,
                next_actions=actions,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Plan output: %s", e)
            return PlanOutput(thought_process="", task_board_updates=[], next_actions=[])

    @staticmethod
    def _parse_reflexion_output(raw: str) -> ReflexionOutput:
        try:
            data = Orchestrator._extract_json(raw)
            status_str = data.get("evaluation_status", "SUCCESS").upper()
            try:
                status = ReflexionStatus(status_str)
            except ValueError:
                status = ReflexionStatus.FAILURE

            new_status_str = data.get("new_status", "COMPLETED")
            try:
                new_status = TaskStatus(new_status_str)
            except ValueError:
                new_status = TaskStatus.COMPLETED

            return ReflexionOutput(
                evaluation_status=status,
                task_id=data.get("task_id", ""),
                new_status=new_status,
                result_summary=data.get("result_summary", ""),
                error_classification=data.get("error_classification", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse Reflexion output: %s", e)
            return ReflexionOutput(
                evaluation_status=ReflexionStatus.SUCCESS,
                result_summary="Parse error",
            )

    @staticmethod
    def _extract_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}

    # ------------------------------------------------------------------
    # Task board management
    # ------------------------------------------------------------------

    @staticmethod
    def _update_task_board(
        task_board: list[TaskBoardItem],
        task_id: str,
        reflexion: ReflexionOutput,
    ) -> None:
        """Apply a reflexion result to the task board."""
        for item in task_board:
            if item.id == task_id:
                item.status = reflexion.new_status
                item.result_summary = reflexion.result_summary
                return
        # If not found, add it
        task_board.append(TaskBoardItem(
            id=task_id,
            action="ADD",
            dependencies=[],
            status=reflexion.new_status,
            result_summary=reflexion.result_summary,
        ))
