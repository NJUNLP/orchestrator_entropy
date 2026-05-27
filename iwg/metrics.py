"""
Evaluation metrics for the IWG benchmark.

Implements the six metrics defined in the paper (Appendix, Section "Metric
Definitions and Calculation Protocols"):

System-Level:
  1. LCS-F1  — Agent sequence structural similarity
  2. TS      — Task Success (all checkpoints matched)

Orchestrator-Level:
  3. Step-SR      — Step Success Rate (micro-average checkpoint accuracy)
  4. EH-F1        — Exception Handling F1
  5. Faithfulness — Context utilization recall
  6. Consistency  — Global trajectory alignment via cosine similarity
"""

from __future__ import annotations

import math
import re
from typing import Optional

from .models import (
    BenchmarkInstance,
    Checkpoint,
    CheckpointType,
    ExecutionStep,
    ExecutionTrajectory,
    NextAction,
)


# ===========================================================================
# System-Level Metrics
# ===========================================================================


def lcs_f1(gold_sequence: list[str], pred_sequence: list[str]) -> dict:
    """Compute LCS-F1: Longest Common Subsequence structural similarity.

    Evaluates the orchestrator's planning logic by comparing the predicted
    sequence of agent calls against the ground truth IWG-synthesized workflow.

    Args:
        gold_sequence: Ground truth agent names [a_1, a_2, ..., a_n].
        pred_sequence: Predicted agent names [â_1, â_2, ..., â_m].

    Returns:
        dict with keys: lcs_length, precision, recall, f1.
    """
    lcs_len = _longest_common_subsequence(gold_sequence, pred_sequence)
    precision = lcs_len / len(pred_sequence) if len(pred_sequence) > 0 else 0.0
    recall = lcs_len / len(gold_sequence) if len(gold_sequence) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"lcs_length": lcs_len, "precision": precision, "recall": recall, "f1": f1}


def task_success(
    gold_checkpoints: list[Checkpoint],
    pred_checkpoint_values: list[str],
) -> float:
    """Compute Task Success (TS): a stringent binary metric.

    A task is considered successful if and only if **all** pre-defined
    checkpoints in the trajectory are correctly matched.

    Returns 1.0 if all checkpoints match, 0.0 otherwise.
    """
    if len(gold_checkpoints) != len(pred_checkpoint_values):
        return 0.0
    for gold, pred in zip(gold_checkpoints, pred_checkpoint_values):
        if not _match_checkpoint(gold, pred):
            return 0.0
    return 1.0


# ===========================================================================
# Orchestrator-Level Metrics
# ===========================================================================


def step_success_rate(
    gold_checkpoints: list[Checkpoint],
    pred_checkpoint_values: list[str],
) -> float:
    """Compute Step Success Rate (Step-SR): micro-average checkpoint accuracy.

    Credits models that partially solve complex tasks by computing the
    ratio of correctly matched checkpoints.
    """
    if not gold_checkpoints:
        return 1.0
    matched = sum(
        1 for g, p in zip(gold_checkpoints, pred_checkpoint_values)
        if _match_checkpoint(g, p)
    )
    # Also count unmatched if pred is shorter
    return matched / len(gold_checkpoints)


def exception_handling_f1(
    gold_recovery_plan: str,
    pred_recovery_action: str,
) -> float:
    """Compute Exception Handling F1 (EH-F1).

    Compares the model's recovery action against the gold recovery plan.
    Uses a classification-based F1 score over recovery strategies:
    Retry, Switch Agent, Abort, Decompose, Ignore.

    Args:
        gold_recovery_plan: The IWG-synthesized gold recovery plan.
        pred_recovery_action: The model's generated recovery action.

    Returns:
        F1 score in [0.0, 1.0].
    """
    gold_class = _classify_recovery_strategy(gold_recovery_plan)
    pred_class = _classify_recovery_strategy(pred_recovery_action)

    if gold_class == pred_class:
        return 1.0
    return 0.0


def faithfulness(
    prev_checkpoint: str,
    current_query: str,
) -> float:
    """Compute Faithfulness: context utilization recall.

    Measures whether the orchestrator correctly utilizes information gained
    from the previous step. Defined as the recall of key entities from the
    previous turn's checkpoint within the current turn's generated query.

    Args:
        prev_checkpoint: The checkpoint value from the previous step (c_{t-1}).
        current_query: The orchestrator's query for the current step (q_t).

    Returns:
        Recall score in [0.0, 1.0].
    """
    cp_tokens = set(_tokenize(prev_checkpoint))
    query_tokens = set(_tokenize(current_query))

    if not cp_tokens:
        return 1.0

    overlap = cp_tokens & query_tokens
    return len(overlap) / len(cp_tokens)


def consistency(
    pred_trajectory_text: str,
    gold_trajectory_text: str,
    embedding_fn: Optional[callable] = None,
) -> float:
    """Compute Consistency: global alignment between predicted and gold trajectories.

    Uses cosine similarity between trajectory embeddings. The paper uses
    text-embedding-v4 model for vector representations.

    Args:
        pred_trajectory_text: The model's full execution log.
        gold_trajectory_text: The IWG-synthesized gold trajectory.
        embedding_fn: Optional function(text) -> list[float] for embeddings.
                      If not provided, uses a simple word-overlap fallback.

    Returns:
        Cosine similarity in [0.0, 1.0].
    """
    if embedding_fn:
        vec_pred = embedding_fn(pred_trajectory_text)
        vec_gold = embedding_fn(gold_trajectory_text)
        return _cosine_similarity(vec_pred, vec_gold)

    # Fallback: Jaccard similarity on word tokens
    pred_tokens = set(_tokenize(pred_trajectory_text))
    gold_tokens = set(_tokenize(gold_trajectory_text))
    if not pred_tokens or not gold_tokens:
        return 0.0
    intersection = pred_tokens & gold_tokens
    union = pred_tokens | gold_tokens
    return len(intersection) / len(union)


# ===========================================================================
# Batch evaluation
# ===========================================================================


def evaluate_trajectory(
    trajectory: ExecutionTrajectory,
    instance: BenchmarkInstance,
    embedding_fn: Optional[callable] = None,
) -> dict[str, float]:
    """Compute all six metrics for a single execution trajectory.

    Returns a dict mapping metric name to value.
    """
    # Extract predicted agent sequence from trajectory steps
    pred_agents = [s.executor_name for s in trajectory.steps if s.executor_name]
    gold_agents = instance.gold_agent_sequence

    # LCS-F1
    lcs_result = lcs_f1(gold_agents, pred_agents)

    # Checkpoint matching
    gold_cps = instance.gold_checkpoints
    pred_cp_values = [
        s.reflexion_output.result_summary if s.reflexion_output else ""
        for s in trajectory.steps
    ]

    # Task Success
    ts = task_success(gold_cps, pred_cp_values[:len(gold_cps)])

    # Step Success Rate
    ssr = step_success_rate(gold_cps, pred_cp_values[:len(gold_cps)])

    # Exception Handling F1 (use first exception if available)
    eh_f1 = 1.0  # default if no exceptions in trajectory
    if instance.exception_scenarios and trajectory.steps:
        gold_plan = instance.gold_recovery_plans.get(
            list(instance.gold_recovery_plans.keys())[0], ""
        )
        pred_action = ""
        for s in trajectory.steps:
            if s.reflexion_output and s.reflexion_output.error_classification:
                pred_action = s.reflexion_output.error_classification
                break
        if gold_plan:
            eh_f1 = exception_handling_f1(gold_plan, pred_action)

    # Faithfulness (average across consecutive steps)
    faith_values = []
    for i in range(1, len(trajectory.steps)):
        prev_cp = gold_cps[i - 1].expected_value if i - 1 < len(gold_cps) else ""
        curr_action = trajectory.steps[i].plan_output.next_actions
        curr_input = curr_action[0].input if curr_action else ""
        faith_values.append(faithfulness(prev_cp, curr_input))
    avg_faith = sum(faith_values) / len(faith_values) if faith_values else 1.0

    # Consistency
    pred_text = " ".join(
        f"{s.plan_output.thought_process} {s.executor_output}"
        for s in trajectory.steps
    )
    gold_text = " ".join(ei.tool_output for ei in instance.environments)
    cons = consistency(pred_text, gold_text, embedding_fn)

    return {
        "LCS-F1": lcs_result["f1"],
        "LCS-Precision": lcs_result["precision"],
        "LCS-Recall": lcs_result["recall"],
        "TaskSuccess": ts,
        "StepSuccessRate": ssr,
        "ExceptionHandlingF1": eh_f1,
        "Faithfulness": avg_faith,
        "Consistency": cons,
    }


# ===========================================================================
# Helpers
# ===========================================================================


def _longest_common_subsequence(seq1: list[str], seq2: list[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(seq1), len(seq2)
    # DP with 1D array for space efficiency
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _match_checkpoint(gold: Checkpoint, pred_value: str) -> bool:
    """Check if a predicted value matches a gold checkpoint."""
    if gold.checkpoint_type == CheckpointType.EXACT_MATCH:
        return gold.expected_value.strip().lower() == pred_value.strip().lower()
    if gold.checkpoint_type == CheckpointType.API_VERIFY:
        # Check for expected pattern in pred
        pattern = gold.expected_value.replace("API(", "").rstrip(")")
        return bool(re.search(re.escape(pattern), pred_value, re.IGNORECASE))
    # Custom 1-shot — simple substring match fallback
    return gold.expected_value.strip().lower() in pred_value.strip().lower()


def _classify_recovery_strategy(text: str) -> str:
    """Classify a recovery action into a strategy category."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["retry", "re-call", "try again", "reattempt"]):
        return "Retry"
    if any(kw in text_lower for kw in ["switch", "different agent", "alternative", "fallback"]):
        return "Switch Agent"
    if any(kw in text_lower for kw in ["abort", "stop", "terminate", "give up"]):
        return "Abort"
    if any(kw in text_lower for kw in ["decompose", "break down", "split"]):
        return "Decompose"
    return "Other"


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
