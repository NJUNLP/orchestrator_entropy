"""
Shared utilities for the decoupled IWG pipeline.

Used by both generate_benchmarks.py (Phase 1) and run_orchestrator.py (Phase 2).
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from openai import OpenAI

from .models import (
    AgentCapability,
    ExecutorDef,
    MASConfig,
)

# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
SEEDS_PATH = os.path.join(PROJECT_DIR, "trajectory-bench", "seeds.json")
BENCH_DIR = os.path.join(PROJECT_DIR, "bench")
TRAJ_DIR = os.path.join(PROJECT_DIR, "trajectories")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_seeds():
    with open(SEEDS_PATH) as f:
        return json.load(f)


# ===========================================================================
# Agent Registry — canonical executor for each capability
# ===========================================================================

AGENT_REGISTRY: dict[AgentCapability, ExecutorDef] = {
    AgentCapability.VISION: ExecutorDef(
        "VisionAgent", [AgentCapability.VISION],
        "Analyzes images, posters, screenshots, receipts, charts and visual documents",
        ["image_recognition", "ocr", "object_detection", "chart_analysis"],
    ),
    AgentCapability.ENTITY_RETRIEVAL: ExecutorDef(
        "EntityRetriever", [AgentCapability.ENTITY_RETRIEVAL],
        "Retrieves structured knowledge about entities (people, places, facts, companies)",
        ["knowledge_base_query", "wiki_lookup", "entity_linking"],
    ),
    AgentCapability.GUI_OPERATION: ExecutorDef(
        "GUIOperator", [AgentCapability.GUI_OPERATION],
        "Performs GUI operations: clicks, form fills, playlist/dashboard updates",
        ["app_operation", "api_call", "form_submission", "dashboard_update"],
    ),
    AgentCapability.FILE_MANAGEMENT: ExecutorDef(
        "FileManagerAgent", [AgentCapability.FILE_MANAGEMENT],
        "Lists, reads, writes, renames, moves files and manages directories",
        ["file_listing", "file_read", "file_write", "directory_navigation", "file_rename"],
    ),
    AgentCapability.STRUCTURED_DATA: ExecutorDef(
        "StructuredDataManager", [AgentCapability.STRUCTURED_DATA],
        "Parses structured files (CSV, JSON, Excel), filters, aggregates and transforms data",
        ["csv_parser", "data_filtering", "statistical_summary", "json_handler"],
    ),
    AgentCapability.TEXT_READING: ExecutorDef(
        "TextReadingAgent", [AgentCapability.TEXT_READING],
        "Reads and extracts content from text documents (PDF, DOCX, TXT, web pages)",
        ["pdf_reader", "text_extraction", "key_point_summarization", "document_parser"],
    ),
    AgentCapability.QUANTITATIVE: ExecutorDef(
        "QuantitativeFinancier", [AgentCapability.QUANTITATIVE],
        "Performs mathematical calculations, financial analysis, statistics, and quantitative reasoning",
        ["ratio_calculation", "math_computation", "financial_modeling", "statistics"],
    ),
    AgentCapability.SUMMARIZATION: ExecutorDef(
        "SummaryAgent", [AgentCapability.SUMMARIZATION],
        "Synthesizes information into comprehensive reports, summaries, and final conclusions",
        ["report_generation", "conclusion_synthesis", "formatting", "content_synthesis"],
    ),
    AgentCapability.CODE_EXECUTION: ExecutorDef(
        "CodeExecutor", [AgentCapability.CODE_EXECUTION],
        "Writes, runs, tests, and debugs code in Python/JavaScript",
        ["python_executor", "code_testing", "debugger", "test_runner"],
    ),
    AgentCapability.WEB_SEARCH: ExecutorDef(
        "WebSearcher", [AgentCapability.WEB_SEARCH],
        "Searches the web for current information, news, papers, and data sources",
        ["web_search", "source_verification", "citation_extraction", "recent_news_lookup"],
    ),
    AgentCapability.AUDIO_PROCESSING: ExecutorDef(
        "AudioMessageAgent", [AgentCapability.AUDIO_PROCESSING],
        "Transcribes and parses audio messages/files to extract user intent and content",
        ["audio_transcription", "fc2_authentication", "file_access", "intent_extraction"],
    ),
}


def get_full_mas(seed_dict: dict | None = None) -> MASConfig:
    """Return the full MAS with ALL available executor agents.

    The Scout Agent receives this complete agent pool and decides which agents
    to use for the task. Seed data does NOT pre-specify capabilities — that is
    the Scout's responsibility (capability-aware inverse analysis).
    """
    max_steps = 20
    if seed_dict:
        max_steps = max(seed_dict.get("expected_steps", 4) + 5, 15)

    return MASConfig(
        executors=list(AGENT_REGISTRY.values()),
        max_steps=max_steps,
        description=f"Full MAS: {len(AGENT_REGISTRY)} agents across all capabilities",
    )


# ===========================================================================
# LLM callables
# ===========================================================================

def create_llm_callable(config: dict) -> Callable[[str], str]:
    """Prompt → response via OpenAI-compatible API."""
    ds = config["openai"]
    client = OpenAI(api_key=ds["api_key"], base_url=ds["base_url"])

    def call(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=ds["model"],
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"
    return call


def create_orchestrator_llm(
    config: dict, model_override: Optional[str] = None
) -> Callable[[str], str]:
    """Create an orchestrator LLM callable supporting multiple backends."""
    model = model_override or config["openai"]["model"]
    model_lower = model.lower()

    if model_lower.startswith("gpt") or (model_lower.startswith("o") and "mini" in model_lower):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for GPT models")
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    elif model_lower.startswith("claude"):
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY required for Claude models")
            client = anthropic.Anthropic(api_key=api_key)

            def call_claude(prompt: str) -> str:
                resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            return call_claude
        except ImportError:
            raise ValueError("pip install anthropic required for Claude models")
    else:
        ds = config["openai"]
        api_key = os.environ.get("ORCH_API_KEY", ds["api_key"])
        base_url = os.environ.get("ORCH_BASE_URL", ds["base_url"])
        client = OpenAI(api_key=api_key, base_url=base_url)

    def call(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"
    return call


# ===========================================================================
# JSON extraction with fallback repair
# ===========================================================================

def extract_json(raw: str) -> dict:
    """Extract JSON from LLM output with progressive fallback repair."""
    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find outermost braces (greedy)
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Repair: truncate trailing garbage, re-close
    repaired = _repair_truncated_json(raw)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # 5. Repair: remove lines with obvious JSON syntax errors
    repaired = _repair_line_by_line(raw)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return {}


def _repair_truncated_json(raw: str) -> Optional[str]:
    """Try to close a truncated JSON string."""
    # Find the last valid structural position
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0

    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                last_valid_pos = i + 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1

    if brace_depth > 0 and last_valid_pos > 0:
        return raw[:last_valid_pos]
    if brace_depth > 0:
        # Try to close remaining braces
        truncated = raw.rstrip().rstrip(',').rstrip()
        if not truncated.endswith('"') and not truncated.endswith(']') and not truncated.endswith('}'):
            # We're mid-value, truncate to last comma
            last_comma = truncated.rfind(',')
            if last_comma > 0:
                truncated = truncated[:last_comma]
        return truncated + '\n}' * brace_depth

    return None


def _repair_line_by_line(raw: str) -> Optional[str]:
    """Remove lines that look broken (unterminated strings, stray characters)."""
    lines = raw.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that look like they're in the middle of a broken string
        if stripped.startswith('"') and not stripped.endswith('"') and not stripped.endswith(','):
            continue
        cleaned.append(line)
    if len(cleaned) < len(lines):
        return '\n'.join(cleaned)
    return None


# ===========================================================================
# Serialization
# ===========================================================================

def serialize_instance(instance) -> dict:
    """Convert BenchmarkInstance to JSON-serializable dict."""
    from .models import BenchmarkInstance
    return {
        "id": instance.id,
        "source": "trajectory-bench/seeds.json",
        "seed_data": {
            "id": instance.seed_data.id,
            "query": instance.seed_data.query,
            "answer": instance.seed_data.answer,
            "domain": instance.seed_data.domain,
            "difficulty": instance.seed_data.difficulty,
            "expected_steps": instance.seed_data.expected_steps,
            "metadata": instance.seed_data.metadata,
        },
        "gold_agent_sequence": instance.gold_agent_sequence,
        "gold_checkpoints": [
            {
                "id": cp.id, "task_mark_id": cp.task_mark_id,
                "type": cp.checkpoint_type.value,
                "expected_value": cp.expected_value,
                "verification_prompt": cp.verification_prompt,
                "step_index": cp.step_index,
            }
            for cp in instance.gold_checkpoints
        ],
        "environments": [
            {
                "id": ei.id, "task_mark_id": ei.task_mark_id,
                "step_index": ei.step_index, "agent_name": ei.agent_name,
                "tool_prompt": ei.tool_prompt, "tool_output": ei.tool_output,
                "checkpoint": {
                    "type": ei.checkpoint.checkpoint_type.value,
                    "expected_value": ei.checkpoint.expected_value,
                    "verification_prompt": ei.checkpoint.verification_prompt,
                } if ei.checkpoint else None,
            }
            for ei in instance.environments
        ],
        "exception_scenarios": instance.exception_scenarios,
        "gold_recovery_plans": instance.gold_recovery_plans,
        "validation_report": {
            "passed": instance.validation_report.passed,
            "tiers": [
                {"tier": r.tier.value, "passed": r.passed, "reasoning": r.reasoning}
                for r in instance.validation_report.results
            ],
        } if instance.validation_report else None,
    }


def serialize_trajectory(traj) -> dict:
    """Convert ExecutionTrajectory to JSON-serializable dict."""
    return {
        "instance_id": traj.instance_id,
        "model_name": traj.model_name,
        "num_steps": len(traj.steps),
        "completed": traj.completed,
        "steps": [
            {
                "step_index": s.step_index,
                "thought_process": s.plan_output.thought_process,
                "actions": [
                    {"agent": a.agent, "task_id": a.task_id, "input": a.input}
                    for a in s.plan_output.next_actions
                ],
                "executor": s.executor_name,
                "executor_output": s.executor_output[:500],
                "entropy": s.scheduling_entropy,
                "scheduling_vector": s.scheduling_vector,
                "reflexion_status": s.reflexion_output.evaluation_status.value
                if s.reflexion_output else None,
                "result_summary": s.reflexion_output.result_summary
                if s.reflexion_output else None,
            }
            for s in traj.steps
        ],
    }
