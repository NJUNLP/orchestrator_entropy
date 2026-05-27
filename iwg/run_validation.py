#!/usr/bin/env python3
"""
Full IWG pipeline validation suite.

Runs the complete Inverse Workflow Generation pipeline on both case studies
from the paper (The White Ribbon + Miss Chris Audio Request). All model
settings are read from iwg/config.json.

Usage:
    python3 iwg/run_validation.py
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iwg.models import (
    AgentCapability,
    BenchmarkInstance,
    Checkpoint,
    CheckpointType,
    MASConfig,
    SeedData,
)
from iwg.scout_agent import ScoutAgent
from iwg.wrapper_agent import WrapperAgent
from iwg.validation import ValidationCommittee, ValidationResult, ValidationTier
from iwg.orchestrator import Orchestrator
from iwg.pipeline import IWGPipeline
from iwg.metrics import (
    evaluate_trajectory,
    lcs_f1,
    task_success,
    step_success_rate,
    exception_handling_f1,
    faithfulness,
    consistency,
)
from iwg.examples import (
    WHITE_RIBBON_MAS,
    WHITE_RIBBON_SEED,
    FINANCE_MAS,
    FINANCE_SEED,
    WHITE_RIBBON_ENVIRONMENTS,
)
from iwg._common import (
    load_config,
    create_llm_callable,
    create_orchestrator_llm,
)


# ===========================================================================
# Test 1: Basic API connectivity
# ===========================================================================

def test_connectivity(config: dict, results: dict):
    """Verify API connectivity and model availability."""
    print("=" * 70)
    print("TEST 1: API Connectivity")
    print("=" * 70)

    ds = config["openai"]
    from openai import OpenAI
    client = OpenAI(api_key=ds["api_key"], base_url=ds["base_url"])

    try:
        response = client.chat.completions.create(
            model=ds["model"],
            messages=[{"role": "user", "content": "Reply with just the word: OK"}],
        )
        msg = response.choices[0].message.content
        print(f"  Status: SUCCESS")
        print(f"  Model: {ds['model']}")
        print(f"  Response: {msg.strip()}")
        print(f"  Usage: {response.usage}")
        results["connectivity"] = {
            "status": "PASS",
            "model": ds["model"],
            "response": msg.strip(),
            "usage": str(response.usage),
        }
        return True
    except Exception as e:
        print(f"  Status: FAILED")
        print(f"  Error: {e}")
        results["connectivity"] = {"status": "FAIL", "error": str(e)}
        return False


# ===========================================================================
# Test 2: Scout Agent — Inverse Planning
# ===========================================================================

def test_scout_agent(config: dict, llm_call, results: dict):
    """Test the Scout Agent with both rule-based and LLM-driven planning."""
    print("\n" + "=" * 70)
    print("TEST 2: Scout Agent — Inverse Planning")
    print("=" * 70)

    scout = ScoutAgent(model_name=config["iwg"]["scout_model"])
    result = {"rule_based": None, "llm_driven": None}

    # 2a: Rule-based
    print("\n  [2a] Rule-based inverse planning")
    print("  " + "-" * 40)
    try:
        plan_rb = scout.plan_rule_based(WHITE_RIBBON_SEED, WHITE_RIBBON_MAS)
        print(f"  Task Marks: {len(plan_rb.task_marks)}")
        for m in plan_rb.task_marks:
            ext = " [EXTENDED]" if m.is_extension else ""
            print(f"    {m.id}: {m.description} → {m.assigned_agent}{ext}")
        result["rule_based"] = {
            "status": "PASS",
            "num_task_marks": len(plan_rb.task_marks),
            "marks": [
                {"id": m.id, "description": m.description, "agent": m.assigned_agent,
                 "capability": m.required_capability.value, "is_extension": m.is_extension}
                for m in plan_rb.task_marks
            ],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        result["rule_based"] = {"status": "FAIL", "error": str(e)}

    # 2b: LLM-driven
    print("\n  [2b] LLM-driven inverse planning")
    print("  " + "-" * 40)
    try:
        # Override the LLM call method
        scout._call_llm = lambda p: llm_call(p)
        plan_llm = scout.plan(WHITE_RIBBON_SEED, WHITE_RIBBON_MAS)
        print(f"  Reasoning: {plan_llm.reasoning[:200]}...")
        print(f"  Task Marks: {len(plan_llm.task_marks)}")
        for m in plan_llm.task_marks:
            print(f"    {m.id}: {m.description} → {m.assigned_agent} (deps={m.dependencies})")
        result["llm_driven"] = {
            "status": "PASS" if plan_llm.task_marks else "PARTIAL",
            "num_task_marks": len(plan_llm.task_marks),
            "reasoning": plan_llm.reasoning[:500],
            "marks": [
                {"id": m.id, "description": m.description, "agent": m.assigned_agent,
                 "capability": m.required_capability.value, "deps": m.dependencies,
                 "checkpoint_hint": m.checkpoint_hint, "is_extension": m.is_extension}
                for m in plan_llm.task_marks
            ],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        result["llm_driven"] = {"status": "FAIL", "error": str(e)}

    results["scout_agent"] = result
    return plan_llm if result["llm_driven"] and result["llm_driven"]["status"] == "PASS" else None


# ===========================================================================
# Test 3: Wrapper Agent — Environment Synthesis
# ===========================================================================

def test_wrapper_agent(config: dict, llm_call, results: dict):
    """Test the Wrapper Agent with both rule-based and LLM-driven synthesis."""
    print("\n" + "=" * 70)
    print("TEST 3: Wrapper Agent — Environment Synthesis")
    print("=" * 70)

    # First get a scout plan
    scout = ScoutAgent()
    plan = scout.plan_rule_based(WHITE_RIBBON_SEED, WHITE_RIBBON_MAS)

    wrapper = WrapperAgent(model_name=config["iwg"]["wrapper_model"])
    result = {"rule_based": None, "llm_driven": None}

    # 3a: Rule-based
    print("\n  [3a] Rule-based environment synthesis")
    print("  " + "-" * 40)
    try:
        wr_output = wrapper.synthesize_rule_based(plan)
        print(f"  Environments: {len(wr_output.environments)}")
        for ei in wr_output.environments:
            cp_val = ei.checkpoint.expected_value if ei.checkpoint else "(none)"
            print(f"    {ei.id} [{ei.agent_name}]: cp='{cp_val}'")
        print(f"  Exception scenarios: {len(wr_output.exception_scenarios)}")
        result["rule_based"] = {
            "status": "PASS",
            "num_environments": len(wr_output.environments),
            "environments": [
                {
                    "id": ei.id,
                    "step_index": ei.step_index,
                    "agent": ei.agent_name,
                    "tool_prompt": ei.tool_prompt[:200],
                    "checkpoint_expected": ei.checkpoint.expected_value if ei.checkpoint else None,
                }
                for ei in wr_output.environments
            ],
            "num_exceptions": len(wr_output.exception_scenarios),
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        result["rule_based"] = {"status": "FAIL", "error": str(e)}

    # 3b: LLM-driven
    print("\n  [3b] LLM-driven environment synthesis")
    print("  " + "-" * 40)
    try:
        wrapper._call_llm = lambda p: llm_call(p)
        wr_llm = wrapper.synthesize(plan)
        print(f"  Environments: {len(wr_llm.environments)}")
        for ei in wr_llm.environments:
            cp_val = ei.checkpoint.expected_value if ei.checkpoint else "(none)"
            print(f"    [{ei.agent_name}]: cp='{cp_val}'")
        result["llm_driven"] = {
            "status": "PASS" if wr_llm.environments else "PARTIAL",
            "num_environments": len(wr_llm.environments),
            "environments": [
                {
                    "step_index": ei.step_index,
                    "agent": ei.agent_name,
                    "tool_prompt": ei.tool_prompt[:200],
                    "tool_output": ei.tool_output[:200],
                    "checkpoint_expected": ei.checkpoint.expected_value if ei.checkpoint else None,
                }
                for ei in wr_llm.environments
            ],
            "num_exceptions": len(wr_llm.exception_scenarios),
            "gold_recovery_plans": wr_llm.gold_recovery_plans,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        result["llm_driven"] = {"status": "FAIL", "error": str(e)}

    results["wrapper_agent"] = result


# ===========================================================================
# Test 4: Validation Committee
# ===========================================================================

def test_validation(config: dict, val_llm_call, results: dict):
    """Test the Three-Tier Validation Committee."""
    print("\n" + "=" * 70)
    print("TEST 4: Validation Committee — Three-Tier Protocol")
    print("=" * 70)

    # Assemble a test instance using the gold-standard White Ribbon environments
    instance = BenchmarkInstance(
        seed_data=WHITE_RIBBON_SEED,
        gold_agent_sequence=["VisionAgent", "EntityRetriever", "EntityRetriever", "GUIOperator"],
        gold_checkpoints=[ei.checkpoint for ei in WHITE_RIBBON_ENVIRONMENTS if ei.checkpoint],
        environments=WHITE_RIBBON_ENVIRONMENTS,
        exception_scenarios=[
            {"step_index": 1, "exception_type": "NetworkTimeout",
             "description": "Entity Retriever API returns 404 timeout"}
        ],
        gold_recovery_plans={
            "NetworkTimeout_step1": "Retry: re-call EntityRetriever with same parameters"
        },
    )

    validator = ValidationCommittee(
        tier1_model=config["iwg"]["validation_tier1_model"],
        tier2_model=config["iwg"]["validation_tier2_model"],
    )
    validator._llm = val_llm_call

    try:
        print("\n  Running three-tier validation...")
        report = validator.validate(instance)
        print(f"  Instance ID: {report.instance_id}")
        print(f"  Overall: {'PASSED' if report.passed else 'FAILED'}")
        for r in report.results:
            print(f"    {r.tier.value} ({r.model_name}): {'PASS' if r.passed else 'FAIL'} "
                  f"— {r.reasoning[:120]}")

        results["validation"] = {
            "status": "PASS" if report.passed else "PARTIAL",
            "overall_passed": report.passed,
            "tiers": [
                {
                    "tier": r.tier.value,
                    "model": r.model_name,
                    "passed": r.passed,
                    "reasoning": r.reasoning[:300],
                }
                for r in report.results
            ],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["validation"] = {"status": "FAIL", "error": str(e)}


# ===========================================================================
# Test 5: Orchestrator MAS — Plan & Reflexion
# ===========================================================================

def test_orchestrator(config: dict, llm_call, results: dict):
    """Test the Orchestrator Plan/Reflexion loop."""
    print("\n" + "=" * 70)
    print("TEST 5: Orchestrator MAS — Plan & Reflexion Loop")
    print("=" * 70)

    orch = Orchestrator(
        mas_config=WHITE_RIBBON_MAS,
        model_name=config["iwg"]["orchestrator_model"],
        llm_callable=llm_call,
    )

    # Build simulated executor that returns gold-standard EI
    env_map = {ei.agent_name: ei for ei in WHITE_RIBBON_ENVIRONMENTS}

    def simulated_executor(agent_name: str, input_prompt: str) -> str:
        ei = env_map.get(agent_name)
        if ei:
            return ei.tool_output
        return f"[Simulated] {agent_name} executed: {input_prompt[:100]}"

    try:
        print(f"\n  Running Orchestrator on: {WHITE_RIBBON_SEED.query}")
        print(f"  Max steps: 10")
        print("  " + "-" * 40)

        trajectory = orch.run(
            user_query=WHITE_RIBBON_SEED.query,
            executor_callable=simulated_executor,
            max_steps=10,
        )

        print(f"\n  Steps executed: {len(trajectory.steps)}")
        print(f"  Completed: {trajectory.completed}")
        print(f"\n  Per-step summary:")
        for s in trajectory.steps:
            print(f"    Step {s.step_index}:")
            print(f"      Plan thought: {s.plan_output.thought_process[:150]}...")
            print(f"      Actions: {[(a.agent, a.task_id) for a in s.plan_output.next_actions]}")
            print(f"      Executor: {s.executor_name}")
            print(f"      Entropy H(t): {s.scheduling_entropy:.4f}")
            print(f"      Scheduling vector: {[round(v, 3) for v in s.scheduling_vector]}")
            if s.reflexion_output:
                print(f"      Reflexion: {s.reflexion_output.evaluation_status.value} "
                      f"— {s.reflexion_output.result_summary[:100]}")

        # Compute metrics
        instance = BenchmarkInstance(
            seed_data=WHITE_RIBBON_SEED,
            gold_agent_sequence=["VisionAgent", "EntityRetriever", "EntityRetriever", "GUIOperator"],
            gold_checkpoints=[ei.checkpoint for ei in WHITE_RIBBON_ENVIRONMENTS if ei.checkpoint],
            environments=WHITE_RIBBON_ENVIRONMENTS,
            exception_scenarios=[],
            gold_recovery_plans={},
        )
        eval_metrics = evaluate_trajectory(trajectory, instance)

        results["orchestrator"] = {
            "status": "PASS" if trajectory.steps else "PARTIAL",
            "num_steps": len(trajectory.steps),
            "completed": trajectory.completed,
            "steps": [
                {
                    "step_index": s.step_index,
                    "thought_process": s.plan_output.thought_process[:300],
                    "actions": [{"agent": a.agent, "task_id": a.task_id, "input": a.input[:150]}
                                for a in s.plan_output.next_actions],
                    "executor": s.executor_name,
                    "executor_output": s.executor_output[:200],
                    "entropy": s.scheduling_entropy,
                    "scheduling_vector": s.scheduling_vector,
                    "reflexion_status": s.reflexion_output.evaluation_status.value if s.reflexion_output else None,
                    "result_summary": s.reflexion_output.result_summary[:200] if s.reflexion_output else None,
                }
                for s in trajectory.steps
            ],
            "metrics": eval_metrics,
        }

        print(f"\n  Metrics:")
        for k, v in eval_metrics.items():
            print(f"    {k}: {v}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["orchestrator"] = {"status": "FAIL", "error": str(e)}


# ===========================================================================
# Test 6: Full IWG Pipeline (End-to-End)
# ===========================================================================

def test_full_pipeline(config: dict, llm_call, val_llm_call, results: dict):
    """Run the complete IWG pipeline end-to-end."""
    print("\n" + "=" * 70)
    print("TEST 6: Full IWG Pipeline (End-to-End)")
    print("=" * 70)

    pipeline = IWGPipeline(mas_config=WHITE_RIBBON_MAS)
    pipeline.configure_llm(llm_call)
    pipeline.configure_validator_llm(val_llm_call)

    try:
        print(f"\n  Seed query: {WHITE_RIBBON_SEED.query}")
        print(f"  Seed answer: {WHITE_RIBBON_SEED.answer}")
        print("\n  Running pipeline: Seed → Scout → Wrapper → Validate → Orchestrate → Evaluate")
        print("  " + "-" * 40)

        # Generate the benchmark instance
        t0 = time.time()
        instance = pipeline.generate(WHITE_RIBBON_SEED)
        t_gen = time.time() - t0
        print(f"\n  [Phase 1] Benchmark generation: {t_gen:.1f}s")
        print(f"    Instance ID: {instance.id}")
        print(f"    Gold agent sequence: {' → '.join(instance.gold_agent_sequence)}")
        print(f"    Checkpoints: {len(instance.gold_checkpoints)}")
        for cp in instance.gold_checkpoints:
            print(f"      {cp.id}: expected='{cp.expected_value}' (type={cp.checkpoint_type.value})")

        # Run orchestrator
        t1 = time.time()
        trajectory = pipeline.run_orchestrator(instance, model_name=config["iwg"]["orchestrator_model"], max_steps=10)
        t_orch = time.time() - t1
        print(f"\n  [Phase 2] Orchestration: {t_orch:.1f}s")
        print(f"    Steps: {len(trajectory.steps)}")
        print(f"    Completed: {trajectory.completed}")

        # Evaluate
        t2 = time.time()
        eval_metrics = pipeline.evaluate(trajectory, instance)
        t_eval = time.time() - t2
        print(f"\n  [Phase 3] Evaluation: {t_eval:.3f}s")
        for k, v in eval_metrics.items():
            print(f"    {k}: {v}")

        results["full_pipeline"] = {
            "status": "PASS" if trajectory.steps else "PARTIAL",
            "total_time_s": round(time.time() - t0, 1),
            "generation_time_s": round(t_gen, 1),
            "orchestration_time_s": round(t_orch, 1),
            "evaluation_time_s": round(t_eval, 3),
            "instance": {
                "id": instance.id,
                "query": instance.seed_data.query,
                "answer": instance.seed_data.answer,
                "gold_agent_sequence": instance.gold_agent_sequence,
                "num_checkpoints": len(instance.gold_checkpoints),
                "checkpoints": [
                    {"id": cp.id, "type": cp.checkpoint_type.value, "expected": cp.expected_value}
                    for cp in instance.gold_checkpoints
                ],
            },
            "trajectory": {
                "num_steps": len(trajectory.steps),
                "completed": trajectory.completed,
                "entropy_values": [s.scheduling_entropy for s in trajectory.steps],
            },
            "metrics": eval_metrics,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results["full_pipeline"] = {"status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}


# ===========================================================================
# Test 7: Metrics computation validation
# ===========================================================================

def test_metrics(results: dict):
    """Validate metric computations with known test cases."""
    print("\n" + "=" * 70)
    print("TEST 7: Metrics Computation Validation")
    print("=" * 70)

    metrics_result = {}

    # LCS-F1 tests
    print("\n  [7a] LCS-F1")
    gold = ["VisionAgent", "EntityRetriever", "EntityRetriever", "GUIOperator"]

    # Perfect match
    perfect = lcs_f1(gold, gold)
    print(f"    Perfect match: F1={perfect['f1']:.4f} (expected 1.0)")
    assert perfect["f1"] == 1.0, f"Expected 1.0, got {perfect['f1']}"

    # Partial match (skip middle step)
    partial = lcs_f1(gold, ["VisionAgent", "EntityRetriever", "GUIOperator"])
    print(f"    Skip M_2: F1={partial['f1']:.4f}, Precision={partial['precision']:.4f}, Recall={partial['recall']:.4f}")
    assert partial["f1"] < 1.0

    # Wrong agent
    wrong = lcs_f1(gold, ["VisionAgent", "GUIOperator", "GUIOperator", "GUIOperator"])
    print(f"    Wrong agents: F1={wrong['f1']:.4f}")

    metrics_result["lcs_f1"] = {
        "perfect": perfect,
        "partial_skip": partial,
        "wrong_agents": wrong,
    }

    # Task Success tests
    print("\n  [7b] Task Success")
    from iwg.metrics import _match_checkpoint

    cps = [ei.checkpoint for ei in WHITE_RIBBON_ENVIRONMENTS if ei.checkpoint]

    # All correct
    correct_vals = ["The White Ribbon", "Michael Haneke", "March 23, 1942",
                    "API(GET /playlist/pl_german_cinema) -> film_tt1345836"]
    ts_all = task_success(cps, correct_vals)
    print(f"    All correct: TS={ts_all} (expected 1.0)")

    # One wrong
    wrong_vals = ["The White Ribbon", "Steven Spielberg", "March 23, 1942",
                  "API(GET /playlist/pl_german_cinema) -> film_tt1345836"]
    ts_wrong = task_success(cps, wrong_vals)
    print(f"    One wrong: TS={ts_wrong} (expected 0.0)")

    metrics_result["task_success"] = {"all_correct": ts_all, "one_wrong": ts_wrong}

    # Step Success Rate
    print("\n  [7c] Step Success Rate")
    ssr_correct = step_success_rate(cps, correct_vals)
    ssr_partial = step_success_rate(cps, wrong_vals)
    print(f"    All correct: Step-SR={ssr_correct:.4f} (expected 1.0)")
    print(f"    3/4 correct: Step-SR={ssr_partial:.4f} (expected 0.75)")

    metrics_result["step_sr"] = {"all_correct": ssr_correct, "three_of_four": ssr_partial}

    # Faithfulness
    print("\n  [7d] Faithfulness")
    f_full = faithfulness("The White Ribbon", "Who directed the film The White Ribbon?")
    f_none = faithfulness("The White Ribbon", "Who directed it?")
    print(f"    Full overlap: Faithfulness={f_full:.4f}")
    print(f"    No overlap: Faithfulness={f_none:.4f}")

    metrics_result["faithfulness"] = {"full_overlap": f_full, "no_overlap": f_none}

    # EH-F1
    print("\n  [7e] Exception Handling F1")
    eh_same = exception_handling_f1("Retry: re-call EntityRetriever", "Retry the entity retriever call")
    eh_diff = exception_handling_f1("Retry: re-call EntityRetriever", "Abort the task completely")
    print(f"    Same strategy (Retry): EH-F1={eh_same} (expected 1.0)")
    print(f"    Different strategy: EH-F1={eh_diff} (expected 0.0)")

    metrics_result["eh_f1"] = {"same_strategy": eh_same, "diff_strategy": eh_diff}

    # Consistency
    print("\n  [7f] Consistency")
    text_a = "The film The White Ribbon was directed by Michael Haneke born in 1942"
    text_b = "Michael Haneke directed The White Ribbon and was born on March 23 1942"
    c_sim = consistency(text_a, text_b)
    print(f"    Similar texts: Consistency={c_sim:.4f}")

    text_c = "Completely unrelated text about different topics"
    c_diff = consistency(text_a, text_c)
    print(f"    Different texts: Consistency={c_diff:.4f}")

    metrics_result["consistency"] = {"similar": c_sim, "different": c_diff}

    results["metrics_validation"] = metrics_result


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("IWG Pipeline — Full Validation Suite")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load config
    config = load_config()
    print(f"\nConfig loaded:")
    print(f"  Model: {config['openai']['model']}")
    print(f"  Base URL: {config['openai']['base_url']}")
    # Create LLM callables
    llm_call = create_llm_callable(config)
    _val_llm = create_llm_callable(config)
    val_llm_call = lambda model_name, prompt: _val_llm(prompt)

    # Results accumulator
    results = {
        "meta": {
            "date": datetime.now().isoformat(),
            "model": config["openai"]["model"],
            "base_url": config["openai"]["base_url"],
        },
    }

    # Run all tests
    tests = []

    # Test 1: Connectivity
    if test_connectivity(config, results):
        tests.append("connectivity")

        # Test 2: Scout Agent
        scout_plan = test_scout_agent(config, llm_call, results)
        tests.append("scout_agent")

        # Test 3: Wrapper Agent
        test_wrapper_agent(config, llm_call, results)
        tests.append("wrapper_agent")

        # Test 4: Validation Committee
        test_validation(config, val_llm_call, results)
        tests.append("validation")

        # Test 5: Orchestrator
        test_orchestrator(config, llm_call, results)
        tests.append("orchestrator")

        # Test 6: Full End-to-End Pipeline
        test_full_pipeline(config, llm_call, val_llm_call, results)
        tests.append("full_pipeline")

    # Test 7: Metrics (always runs, no API needed)
    test_metrics(results)
    tests.append("metrics_validation")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "validation_results.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print("\n\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    def print_status(name, test, indent=0):
        prefix = "  " * indent
        if isinstance(test, dict):
            if "status" in test:
                s = test["status"]
                icon = "[OK]" if s == "PASS" else "[~] " if s == "PARTIAL" else "[FAIL]"
                print(f"{prefix}{icon} {name}: {s}")
            else:
                # Nested results
                has_children = any(isinstance(v, dict) and "status" in v for v in test.values())
                if has_children:
                    print(f"{prefix}{name}:")
                    for sub_name, sub_test in test.items():
                        if isinstance(sub_test, dict) and "status" in sub_test:
                            print_status(sub_name, sub_test, indent + 1)

    for test_name in ["connectivity", "scout_agent", "wrapper_agent", "validation",
                       "orchestrator", "full_pipeline", "metrics_validation"]:
        if test_name in results:
            print_status(test_name, results[test_name])

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
