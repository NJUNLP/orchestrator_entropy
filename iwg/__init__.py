"""
Inverse Workflow Generation (IWG) — A multi-agent pipeline for synthesizing
process-verifiable, high-complexity benchmarks with dense intermediate checkpoints.

Based on: "Recognize Your Orchestrator: An Entropy Dynamics Perspective
for LLM Multi-Agent Systems" (ICML 2026).

Pipeline:
  1. Scout Agent     — Inverse planning: QA pair → Task Mark DAG
  2. Wrapper Agent   — Environment synthesis: Task Marks → EI + Checkpoints
  3. Validation      — Three-tier quality control
  4. Orchestrator    — Plan/Reflexion MAS execution
  5. Metrics         — LCS-F1, TS, Step-SR, EH-F1, Faithfulness, Consistency
"""

from .models import (
    AgentCapability,
    BenchmarkInstance,
    Checkpoint,
    CheckpointType,
    EnvironmentInfo,
    ExecutionStep,
    ExecutionTrajectory,
    ExecutorDef,
    MASConfig,
    NextAction,
    PlanOutput,
    ReflexionOutput,
    ReflexionStatus,
    ScoutPlan,
    SeedData,
    TaskBoardItem,
    TaskMark,
    TaskStatus,
    ValidationReport,
    ValidationResult,
    ValidationTier,
    WrapperOutput,
)
from .scout_agent import ScoutAgent
from .wrapper_agent import WrapperAgent
from .validation import ValidationCommittee
from .orchestrator import Orchestrator
from .pipeline import IWGPipeline
from . import metrics

__all__ = [
    # Pipeline
    "IWGPipeline",
    # Agents
    "ScoutAgent",
    "WrapperAgent",
    "ValidationCommittee",
    "Orchestrator",
    # Models
    "AgentCapability",
    "BenchmarkInstance",
    "Checkpoint",
    "CheckpointType",
    "EnvironmentInfo",
    "ExecutionStep",
    "ExecutionTrajectory",
    "ExecutorDef",
    "MASConfig",
    "NextAction",
    "PlanOutput",
    "ReflexionOutput",
    "ReflexionStatus",
    "ScoutPlan",
    "SeedData",
    "TaskBoardItem",
    "TaskMark",
    "TaskStatus",
    "ValidationReport",
    "ValidationResult",
    "ValidationTier",
    "WrapperOutput",
    # Metrics
    "metrics",
]
