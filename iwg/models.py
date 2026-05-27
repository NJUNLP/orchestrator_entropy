"""
Core data models for the Inverse Workflow Generation (IWG) pipeline.

The IWG pipeline synthesizes process-verifiable, high-complexity benchmarks
by reconstructing the necessary environment states and tool outputs backward
from a target solution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Base / enumerations
# ---------------------------------------------------------------------------


class AgentCapability(str, Enum):
    """Capabilities an executor agent can expose."""
    VISION = "vision"
    ENTITY_RETRIEVAL = "entity_retrieval"
    GUI_OPERATION = "gui_operation"
    FILE_MANAGEMENT = "file_management"
    STRUCTURED_DATA = "structured_data"
    TEXT_READING = "text_reading"
    QUANTITATIVE = "quantitative"
    SUMMARIZATION = "summarization"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    AUDIO_PROCESSING = "audio_processing"


class CheckpointType(str, Enum):
    EXACT_MATCH = "exact_match"
    API_VERIFY = "api_verify"
    CUSTOM_1SHOT = "custom_1shot"


class ValidationTier(str, Enum):
    TIER_1 = "tier_1"  # open-source solvability
    TIER_2 = "tier_2"  # proprietary consistency
    TIER_3 = "tier_3"  # human-in-the-loop


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ReflexionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


# ---------------------------------------------------------------------------
# MAS configuration
# ---------------------------------------------------------------------------


@dataclass
class ExecutorDef:
    """Definition of an executor agent in the target MAS."""
    name: str
    capabilities: list[AgentCapability]
    description: str = ""
    tools: list[str] = field(default_factory=list)


@dataclass
class MASConfig:
    """Target Multi-Agent System configuration."""
    executors: list[ExecutorDef]
    max_steps: int = 20
    description: str = ""


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------


@dataclass
class SeedData:
    """High-quality, pre-verified QA pair used as the seed for IWG."""
    query: str
    answer: str
    domain: str = "general"
    metadata: dict = field(default_factory=dict)
    id: str = ""
    difficulty: str = "medium"
    expected_steps: int = 3


# ---------------------------------------------------------------------------
# Scout outputs
# ---------------------------------------------------------------------------


@dataclass
class TaskMark:
    """A single task mark produced by the Scout Agent.

    Represents one atomic step that must be performed by a specific executor
    to move from the current state toward the final answer.
    """
    id: str = field(default_factory=lambda: f"M_{uuid.uuid4().hex[:6]}")
    description: str = ""
    assigned_agent: str = ""         # executor name
    required_capability: AgentCapability = AgentCapability.ENTITY_RETRIEVAL
    dependencies: list[str] = field(default_factory=list)  # ids of prerequisite TaskMarks
    checkpoint_hint: str = ""        # what should be verified at this step
    is_extension: bool = False       # True if this step was added by Scout beyond the seed


@dataclass
class ScoutPlan:
    """Output of the Scout Agent — a DAG of TaskMarks."""
    seed_data: SeedData
    task_marks: list[TaskMark]
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Wrapper outputs
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A verification checkpoint embedded in the synthesized environment."""
    id: str = field(default_factory=lambda: f"CP_{uuid.uuid4().hex[:6]}")
    task_mark_id: str = ""
    checkpoint_type: CheckpointType = CheckpointType.EXACT_MATCH
    expected_value: str = ""
    verification_prompt: str = ""
    step_index: int = 0


@dataclass
class EnvironmentInfo:
    """Synthesized environmental information for one task mark.

    This is what the executor agent would observe as tool output / context
    when executing its assigned task.
    """
    id: str = field(default_factory=lambda: f"EI_{uuid.uuid4().hex[:6]}")
    task_mark_id: str = ""
    step_index: int = 0
    agent_name: str = ""
    tool_prompt: str = ""            # the prompt/query the agent receives
    tool_output: str = ""            # synthesized tool response / observation
    checkpoint: Optional[Checkpoint] = None


@dataclass
class WrapperOutput:
    """Output of the Wrapper Agent — a fully instantiated interactive environment."""
    scout_plan: ScoutPlan
    environments: list[EnvironmentInfo]
    exception_scenarios: list[dict] = field(default_factory=list)
    gold_recovery_plans: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    tier: ValidationTier
    passed: bool
    model_name: str = ""
    reasoning: str = ""
    score: float = 0.0


@dataclass
class ValidationReport:
    instance_id: str
    results: list[ValidationResult]
    passed: bool = False

    @property
    def is_fully_validated(self) -> bool:
        return self.passed and len(self.results) >= 3


# ---------------------------------------------------------------------------
# Orchestrator MAS types
# ---------------------------------------------------------------------------


@dataclass
class TaskBoardItem:
    """A task on the Orchestrator's DAG task board."""
    id: str
    action: str  # ADD | MODIFY
    dependencies: list[str]
    status: TaskStatus = TaskStatus.PENDING
    result_summary: str = ""


@dataclass
class NextAction:
    """A scheduled next action for an executor."""
    agent: str
    input: str
    task_id: str = ""


@dataclass
class PlanOutput:
    """Orchestrator Plan Mode output."""
    thought_process: str
    task_board_updates: list[TaskBoardItem]
    next_actions: list[NextAction]


@dataclass
class ReflexionOutput:
    """Orchestrator Reflexion Mode output."""
    evaluation_status: ReflexionStatus
    task_id: str = ""
    new_status: TaskStatus = TaskStatus.PENDING
    result_summary: str = ""
    error_classification: str = ""


# ---------------------------------------------------------------------------
# Complete benchmark instance
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkInstance:
    """A complete IWG-synthesized benchmark instance ready for evaluation."""
    seed_data: SeedData
    gold_agent_sequence: list[str]        # ordered list of agent names
    gold_checkpoints: list[Checkpoint]    # ordered checkpoints
    environments: list[EnvironmentInfo]
    exception_scenarios: list[dict]
    gold_recovery_plans: dict[str, str]
    validation_report: Optional[ValidationReport] = None
    id: str = field(default_factory=lambda: f"IWG_{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------------------
# Execution trajectory (collected at runtime)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionStep:
    """One step of an actual MAS execution."""
    step_index: int
    plan_output: PlanOutput
    reflexion_output: Optional[ReflexionOutput] = None
    executor_name: str = ""
    executor_output: str = ""
    scheduling_entropy: float = 0.0
    scheduling_vector: list[float] = field(default_factory=list)


@dataclass
class ExecutionTrajectory:
    """Complete execution trajectory of an orchestrator on a benchmark instance."""
    instance_id: str
    model_name: str
    steps: list[ExecutionStep] = field(default_factory=list)
    completed: bool = False
    matched_checkpoints: list[bool] = field(default_factory=list)
