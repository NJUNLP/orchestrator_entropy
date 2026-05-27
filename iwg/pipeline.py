"""
IWG Pipeline — Main Orchestrator.

Wires the full Inverse Workflow Generation pipeline:
  1. Seed Data → Scout Agent → Task Marks (DAG)
  2. Task Marks → Wrapper Agent → Environment Info + Checkpoints
  3. Env Info → Validation Committee → Quality-controlled Benchmark Instance
  4. Benchmark Instance → Orchestrator MAS → Execution Trajectory
  5. Trajectory → Metrics → Performance Report

This pipeline produces the dense, step-level observation checkpoints required
to fit the mean-field entropy dynamics equation (Section 2 of the paper).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .models import (
    BenchmarkInstance,
    ExecutionTrajectory,
    MASConfig,
    ScoutPlan,
    SeedData,
    WrapperOutput,
)
from .scout_agent import ScoutAgent
from .wrapper_agent import WrapperAgent
from .validation import ValidationCommittee
from .orchestrator import Orchestrator
from .metrics import evaluate_trajectory

logger = logging.getLogger(__name__)


class IWGPipeline:
    """Complete Inverse Workflow Generation pipeline.

    Usage:
        pipeline = IWGPipeline(mas_config)
        pipeline.configure_llm(my_llm_callable)

        # Generate a benchmark instance from seed data
        instance = pipeline.generate(seed_data)

        # Run an orchestrator against the instance
        trajectory = pipeline.run_orchestrator(instance)

        # Evaluate the trajectory
        report = pipeline.evaluate(trajectory, instance)
    """

    def __init__(
        self,
        mas_config: MASConfig,
        scout_model: str = "",
        wrapper_model: str = "",
    ):
        self.mas_config = mas_config
        self.scout = ScoutAgent(model_name=scout_model)
        self.wrapper = WrapperAgent(model_name=wrapper_model)
        self.validator = ValidationCommittee()
        self._llm_callable: Optional[Callable[[str], str]] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_llm(self, callable_: Callable[[str], str]) -> None:
        """Inject an LLM callable for all agents.

        Args:
            callable_: Function(prompt: str) -> str that calls the LLM API.
        """
        self._llm_callable = callable_

    def configure_validator_llm(
        self, callable_: Callable[[str, str], str]
    ) -> None:
        """Inject an LLM callable for the validation committee.

        Args:
            callable_: Function(model_name: str, prompt: str) -> str.
        """
        self.validator._llm = callable_

    def configure_human_review(
        self, callback: Callable[[str], str]
    ) -> None:
        """Set the human-in-the-loop callback for Tier 3 validation."""
        self.validator.tier3_callback = callback

    # ------------------------------------------------------------------
    # Step 1: Generate benchmark instance from seed data
    # ------------------------------------------------------------------

    def generate(self, seed: SeedData) -> BenchmarkInstance:
        """Synthesize a complete benchmark instance from seed data.

        Runs the full IWG pipeline:
          Scout → Wrapper → Validation

        Args:
            seed: Pre-verified QA pair.

        Returns:
            A validated BenchmarkInstance ready for orchestrator evaluation.
        """
        logger.info("IWG Pipeline: generating instance from '%s'", seed.query)

        # Phase 1: Scout — Inverse Planning
        scout_plan = self._run_scout(seed)

        # Phase 2: Wrapper — Environment Synthesis
        wrapper_output = self._run_wrapper(scout_plan)

        # Phase 3: Assemble instance
        instance = self._assemble_instance(seed, scout_plan, wrapper_output)

        # Phase 4: Validation Committee
        if self.validator._llm or self.validator.tier3_callback:
            self.validator.validate(instance)

        return instance

    def generate_batch(self, seeds: list[SeedData]) -> list[BenchmarkInstance]:
        """Generate benchmark instances from multiple seeds."""
        instances = [self.generate(seed) for seed in seeds]
        return self.validator.validate_batch(instances)

    # ------------------------------------------------------------------
    # Step 2: Run orchestrator against an instance
    # ------------------------------------------------------------------

    def run_orchestrator(
        self,
        instance: BenchmarkInstance,
        model_name: str = "",
        max_steps: int = 20,
    ) -> ExecutionTrajectory:
        """Execute the orchestrator MAS on a benchmark instance.

        The orchestrator interacts with the synthesized environment,
        producing per-step scheduling vectors p_k that are used to
        compute the scheduling entropy H(t).

        Args:
            instance: The benchmark instance to run.
            model_name: The LLM model to use as orchestrator.
            max_steps: Maximum orchestration steps.

        Returns:
            Complete execution trajectory with per-step entropy values.
        """
        orch = Orchestrator(
            mas_config=self.mas_config,
            model_name=model_name,
            llm_callable=self._llm_callable,
        )

        # Build a simulated executor that returns the synthesized EI
        env_map = {ei.agent_name: ei for ei in instance.environments}

        def simulated_executor(agent_name: str, input_prompt: str) -> str:
            ei = env_map.get(agent_name)
            if ei:
                return ei.tool_output
            return f"[Simulated] {agent_name} executed: {input_prompt[:100]}"

        # Build checkpoint validator
        cp_map = {cp.step_index: cp for cp in instance.gold_checkpoints}

        def checkpoint_validator(expected: str, actual: str) -> bool:
            return expected.strip().lower() in actual.strip().lower()

        trajectory = orch.run(
            user_query=instance.seed_data.query,
            executor_callable=simulated_executor,
            max_steps=max_steps,
            checkpoint_validator=checkpoint_validator,
        )
        trajectory.instance_id = instance.id

        return trajectory

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trajectory: ExecutionTrajectory,
        instance: BenchmarkInstance,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> dict:
        """Compute all six evaluation metrics for a trajectory.

        Returns a dict with all metric values and a summary.
        """
        metrics = evaluate_trajectory(trajectory, instance, embedding_fn)

        # Add entropy-dynamics related information
        entropy_values = [s.scheduling_entropy for s in trajectory.steps]
        metrics["mean_entropy"] = (
            sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
        )
        metrics["max_entropy"] = max(entropy_values) if entropy_values else 0.0
        metrics["num_steps"] = len(trajectory.steps)
        metrics["completed"] = float(trajectory.completed)

        return metrics

    # ------------------------------------------------------------------
    # Internal: pipeline phases
    # ------------------------------------------------------------------

    def _run_scout(self, seed: SeedData) -> ScoutPlan:
        """Run Scout Agent — use LLM if configured, else rule-based."""
        if self._llm_callable:
            # Override the scout's LLM call method temporarily
            original_call = self.scout._call_llm
            self.scout._call_llm = lambda p: self._llm_callable(p)
            try:
                return self.scout.plan(seed, self.mas_config)
            finally:
                self.scout._call_llm = original_call
        else:
            logger.info("Scout: using rule-based planning (no LLM configured)")
            return self.scout.plan_rule_based(seed, self.mas_config)

    def _run_wrapper(self, scout_plan: ScoutPlan) -> WrapperOutput:
        """Run Wrapper Agent — use LLM if configured, else rule-based."""
        if self._llm_callable:
            original_call = self.wrapper._call_llm
            self.wrapper._call_llm = lambda p: self._llm_callable(p)
            try:
                return self.wrapper.synthesize(scout_plan)
            finally:
                self.wrapper._call_llm = original_call
        else:
            logger.info("Wrapper: using rule-based synthesis (no LLM configured)")
            return self.wrapper.synthesize_rule_based(scout_plan)

    @staticmethod
    def _assemble_instance(
        seed: SeedData,
        scout_plan: ScoutPlan,
        wrapper_output: WrapperOutput,
    ) -> BenchmarkInstance:
        """Combine Scout + Wrapper outputs into a BenchmarkInstance."""
        gold_agents = [m.assigned_agent for m in scout_plan.task_marks]
        gold_checkpoints = [
            ei.checkpoint for ei in wrapper_output.environments
            if ei.checkpoint is not None
        ]

        return BenchmarkInstance(
            seed_data=seed,
            gold_agent_sequence=gold_agents,
            gold_checkpoints=gold_checkpoints,
            environments=wrapper_output.environments,
            exception_scenarios=wrapper_output.exception_scenarios,
            gold_recovery_plans=wrapper_output.gold_recovery_plans,
        )
