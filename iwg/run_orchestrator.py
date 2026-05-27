#!/usr/bin/env python3
"""
Phase 2: Orchestrator Execution (Plan/Reflexion Loop).

Loads a pre-generated benchmark from bench/, runs the orchestrator with any
model, and saves trajectory + metrics to trajectories/ separately.

Completely decoupled from generate_benchmarks.py (Phase 1).
Multiple models can run against the SAME benchmark for comparison.

Usage:
    # Run one benchmark with one model
    python3 iwg/run_orchestrator.py --bench 001 --model <model_name>

    # Run all benchmarks with one model
    python3 iwg/run_orchestrator.py --all --model <model_name>

    # Run one benchmark with multiple models (comparison)
    python3 iwg/run_orchestrator.py --bench 001 --model <model_a>,<model_b>

    # List available benchmarks
    python3 iwg/run_orchestrator.py --list
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iwg.models import MASConfig, ExecutorDef, AgentCapability
from iwg.orchestrator import Orchestrator
from iwg.metrics import evaluate_trajectory
from iwg._common import (
    load_config, create_orchestrator_llm, extract_json,
    serialize_trajectory, BENCH_DIR, TRAJ_DIR,
)

# Reconstruct MAS from a bench file's gold_agent_sequence
# (This avoids needing to re-read the seed — the bench is self-contained)

def _reconstruct_mas(instance_data: dict) -> MASConfig:
    """Reconstruct a minimal MASConfig from the benchmark instance data.

    The benchmark stores environment info with agent names and capabilities.
    We rebuild just enough MAS for the orchestrator to work.
    """
    envs = instance_data.get("environments", [])
    seen: dict[str, set[str]] = {}
    for ei in envs:
        name = ei.get("agent_name", "")
        if name not in seen:
            seen[name] = set()

    # Build executors from what we know
    executors = []
    for name in instance_data.get("gold_agent_sequence", []):
        if name not in [e.name for e in executors]:
            # Infer capability from the checkpoint type
            cap = AgentCapability.ENTITY_RETRIEVAL  # default
            for ei in envs:
                if ei.get("agent_name") == name and ei.get("checkpoint"):
                    cp_type = ei["checkpoint"].get("type", "")
                    if cp_type == "exact_match":
                        cap = AgentCapability.ENTITY_RETRIEVAL
                    break
            executors.append(ExecutorDef(
                name=name,
                capabilities=[cap],
                description=f"Agent from benchmark: {name}",
                tools=[],
            ))

    if not executors:
        executors = [
            ExecutorDef("EntityRetriever", [AgentCapability.ENTITY_RETRIEVAL],
                        "Default agent", ["knowledge_base_query"]),
        ]

    return MASConfig(
        executors=executors,
        max_steps=20,
        description=f"Reconstructed MAS from benchmark ({len(executors)} agents)",
    )


def load_benchmark(bench_path: str) -> dict:
    """Load a benchmark instance from a bench/*.json file."""
    with open(bench_path) as f:
        return json.load(f)


def list_benchmarks() -> list[str]:
    """List all benchmark files in bench/, sorted by index."""
    if not os.path.isdir(BENCH_DIR):
        return []
    files = [f for f in os.listdir(BENCH_DIR) if f.endswith(".json") and not f.startswith("_")]
    files.sort()
    return files


def run_one(bench_data: dict, model_name: str, orch_llm) -> dict:
    """Run orchestrator on one benchmark instance, return trajectory + metrics."""
    instance_data = bench_data["instance"]
    seed = instance_data["seed_data"]
    instance_id = instance_data["id"]

    print(f"\n  Query: {seed['query'][:100]}...")
    print(f"  Gold agents: {' → '.join(instance_data['gold_agent_sequence'])}")
    print(f"  Checkpoints: {len(instance_data['gold_checkpoints'])}")

    mas = _reconstruct_mas(instance_data)
    orch = Orchestrator(mas_config=mas, model_name=model_name, llm_callable=orch_llm)

    # Build simulated executor from benchmark environments
    env_map = {ei["agent_name"]: ei for ei in instance_data["environments"]}

    def simulated_executor(agent_name: str, input_prompt: str) -> str:
        ei = env_map.get(agent_name)
        if ei:
            return ei.get("tool_output", f"[Simulated] {agent_name}: no output")
        return f"[Simulated] {agent_name} executed: {input_prompt[:150]}..."

    t0 = time.time()
    trajectory = orch.run(
        user_query=seed["query"],
        executor_callable=simulated_executor,
        max_steps=mas.max_steps,
    )
    trajectory.instance_id = instance_id
    t_orch = time.time() - t0

    # Reconstruct checkpoint objects for metrics
    from iwg.models import Checkpoint, CheckpointType, BenchmarkInstance, SeedData, EnvironmentInfo
    gold_cps = []
    for cp in instance_data["gold_checkpoints"]:
        try:
            cpt = CheckpointType(cp["type"])
        except (ValueError, KeyError):
            cpt = CheckpointType.EXACT_MATCH
        gold_cps.append(Checkpoint(
            id=cp.get("id", ""),
            task_mark_id=cp.get("task_mark_id", ""),
            checkpoint_type=cpt,
            expected_value=cp.get("expected_value", ""),
            verification_prompt=cp.get("verification_prompt", ""),
            step_index=cp.get("step_index", 0),
        ))

    dummy_instance = BenchmarkInstance(
        seed_data=SeedData(query=seed["query"], answer=seed["answer"],
                           domain=seed.get("domain", "general")),
        gold_agent_sequence=instance_data["gold_agent_sequence"],
        gold_checkpoints=gold_cps,
        environments=[],
        exception_scenarios=instance_data.get("exception_scenarios", []),
        gold_recovery_plans=instance_data.get("gold_recovery_plans", {}),
    )

    metrics = evaluate_trajectory(trajectory, dummy_instance)
    metrics["orchestration_time_s"] = round(t_orch, 1)
    metrics["num_steps"] = len(trajectory.steps)

    print(f"  ✓ {len(trajectory.steps)} steps in {t_orch:.1f}s")
    print(f"    LCS-F1={metrics['LCS-F1']:.3f}  TS={metrics['TaskSuccess']:.1f}  "
          f"StepSR={metrics['StepSuccessRate']:.3f}  Faith={metrics['Faithfulness']:.3f}  "
          f"EH-F1={metrics['ExceptionHandlingF1']:.1f}")

    return {
        "model_name": model_name,
        "trajectory": serialize_trajectory(trajectory),
        "metrics": metrics,
    }


def save_trajectory(bench_data: dict, orch_result: dict) -> str:
    """Save orchestration trajectory to trajectories/."""
    os.makedirs(TRAJ_DIR, exist_ok=True)

    instance_id = bench_data["instance"]["id"]
    model = orch_result["model_name"]
    bench_index = bench_data.get("index", 0)

    filename = f"{bench_index:03d}_{instance_id}_{model}.json"
    filepath = os.path.join(TRAJ_DIR, filename)

    output = {
        "benchmark_id": instance_id,
        "benchmark_file": bench_data.get("file", ""),
        "model_name": model,
        "run_at": datetime.now().isoformat(),
        "phase": "orchestration",
        "metrics": orch_result["metrics"],
        "trajectory": orch_result["trajectory"],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="IWG Phase 2: Orchestrator Execution")
    parser.add_argument("--bench", type=str, help="Benchmark index (e.g., 001) or file path")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks in bench/")
    parser.add_argument("--model", type=str, default="",
                        help="Model name(s), comma-separated for multi-model comparison")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        files = list_benchmarks()
        print(f"Benchmarks in {BENCH_DIR}/ ({len(files)} files):")
        for f in files:
            # Show a quick preview
            data = load_benchmark(os.path.join(BENCH_DIR, f))
            seed = data["instance"]["seed_data"]
            print(f"  {f}")
            print(f"    id={seed['id']}  query={seed['query'][:80]}...")
        return

    if not args.bench and not args.all:
        parser.print_help()
        print("\nExamples:")
        print("  python3 iwg/run_orchestrator.py --list")
        print("  python3 iwg/run_orchestrator.py --bench 001 --model <model_name>")
        print("  python3 iwg/run_orchestrator.py --all --model <model_name>")
        print("  python3 iwg/run_orchestrator.py --bench 001 --model <model_a>,<model_b>")
        sys.exit(1)

    config = load_config()
    models = [m.strip() for m in args.model.split(",")]

    print("=" * 70)
    print("IWG Phase 2: Orchestrator Execution (Plan/Reflexion Loop)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Models: {models}")
    print(f"Output: {TRAJ_DIR}/")
    print("=" * 70)

    # Determine which benchmarks to run
    bench_files = []
    if args.bench:
        # Try as index first, then as file path
        if args.bench.isdigit():
            idx = int(args.bench)
            pattern = f"{idx:03d}_"
            matches = [f for f in list_benchmarks() if f.startswith(pattern)]
            bench_files = [os.path.join(BENCH_DIR, m) for m in matches]
        elif os.path.exists(args.bench):
            bench_files = [args.bench]
        else:
            # Try matching by seed_id
            matches = [f for f in list_benchmarks() if args.bench in f]
            bench_files = [os.path.join(BENCH_DIR, m) for m in matches]

        if not bench_files:
            print(f"ERROR: No benchmark found for '{args.bench}'")
            sys.exit(1)
    else:
        bench_files = [os.path.join(BENCH_DIR, f) for f in list_benchmarks()]

    print(f"\nBenchmarks: {len(bench_files)}")
    for bf in bench_files:
        print(f"  {os.path.basename(bf)}")

    # Run each benchmark × each model
    all_results = []
    for bf in bench_files:
        bname = os.path.basename(bf)
        print(f"\n{'#'*60}")
        print(f"# Loading: {bname}")
        print(f"{'#'*60}")

        bench_data = load_benchmark(bf)
        bench_data["file"] = bname

        for model in models:
            print(f"\n  [{model}] Running orchestrator...")
            try:
                orch_llm = create_orchestrator_llm(config, model)
            except ValueError as e:
                print(f"  ⚠ Cannot use {model}: {e}")
                continue

            try:
                result = run_one(bench_data, model, orch_llm)
                filepath = save_trajectory(bench_data, result)
                print(f"  💾 Saved: trajectories/{os.path.basename(filepath)}")
                all_results.append({
                    "benchmark": bname,
                    "model": model,
                    "status": "success",
                    "trajectory_file": os.path.basename(filepath),
                    "metrics": result["metrics"],
                })
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    "benchmark": bname, "model": model,
                    "status": "error", "error": str(e),
                })

    # Summary
    print("\n\n" + "=" * 70)
    print("ORCHESTRATION COMPLETE")
    print("=" * 70)
    for r in all_results:
        icon = "✓" if r["status"] == "success" else "❌"
        if r["status"] == "success":
            m = r["metrics"]
            print(f"  {icon} {r['benchmark']} @ {r['model']}: "
                  f"LCS-F1={m['LCS-F1']:.3f} TS={m['TaskSuccess']:.1f} "
                  f"Faith={m['Faithfulness']:.3f} → {r['trajectory_file']}")
        else:
            print(f"  {icon} {r['benchmark']} @ {r['model']}: {r['error']}")

    # Cross-model comparison (if multiple models)
    if len(models) > 1:
        print("\n  Cross-model comparison:")
        by_bench = {}
        for r in all_results:
            if r["status"] != "success":
                continue
            by_bench.setdefault(r["benchmark"], {})[r["model"]] = r["metrics"]
        for bench, models_metrics in by_bench.items():
            print(f"  {bench}:")
            for model, m in models_metrics.items():
                print(f"    {model}: LCS-F1={m['LCS-F1']:.3f}  "
                      f"TS={m['TaskSuccess']:.1f}  Faith={m['Faithfulness']:.3f}  "
                      f"Consistency={m['Consistency']:.3f}")


if __name__ == "__main__":
    main()
