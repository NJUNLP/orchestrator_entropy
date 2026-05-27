#!/usr/bin/env python3
"""
Phase 1: Benchmark Generation (IWG Pipeline).

Pure data generation — NO orchestration. Reads seeds from trajectory-bench/,
runs Scout→Wrapper→Validation, and saves static benchmark files to bench/.

Completely decoupled from run_orchestrator.py (Phase 2).

Usage:
    python3 iwg/generate_benchmarks.py                  # all seeds
    python3 iwg/generate_benchmarks.py --seed-id seed_001  # single seed
    python3 iwg/generate_benchmarks.py --resume            # skip existing
    python3 iwg/generate_benchmarks.py --skip-validation   # faster generation
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iwg.models import BenchmarkInstance, SeedData
from iwg.scout_agent import ScoutAgent
from iwg.wrapper_agent import WrapperAgent
from iwg.validation import ValidationCommittee
from iwg._common import (
    load_config, load_seeds, get_full_mas,
    create_llm_callable, extract_json,
    serialize_instance, BENCH_DIR,
)


def generate_one(seed_dict: dict, llm_call, val_llm_call, index: int, total: int,
                 skip_validation: bool = False, model_name: str = ""):
    """Run Scout→Wrapper→Validation for one seed. Returns BenchmarkInstance or None."""
    seed_id = seed_dict["id"]
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] Seed: {seed_id}")
    print(f"  Query: {seed_dict['query'][:120]}...")
    print(f"  Answer: {seed_dict['answer'][:80]}...")
    print(f"  Domain: {seed_dict.get('domain')} | Difficulty: {seed_dict.get('difficulty')} "
          f"| Expected steps: {seed_dict.get('expected_steps', '?')}")

    mas = get_full_mas(seed_dict)
    print(f"  Full MAS: {len(mas.executors)} agents available — Scout will select which to use "
          f"(max_steps={mas.max_steps})")
    print(f"{'='*60}")

    seed = SeedData(
        id=seed_id,
        query=seed_dict["query"],
        answer=seed_dict["answer"],
        domain=seed_dict.get("domain", "general"),
        difficulty=seed_dict.get("difficulty", "medium"),
        expected_steps=seed_dict.get("expected_steps", 3),
        metadata=seed_dict.get("metadata", {}),
    )

    t_start = time.time()

    # --- Scout ---
    print("\n  [Scout] Inverse planning...")
    scout = ScoutAgent(model_name=model_name)
    scout._call_llm = lambda p: llm_call(p)
    t0 = time.time()
    plan = scout.plan(seed, mas)
    print(f"  ✓ {len(plan.task_marks)} task marks ({time.time() - t0:.1f}s)")
    for m in plan.task_marks:
        ext = " [EXTENDED]" if m.is_extension else ""
        print(f"      {m.id}: {m.description[:80]} → {m.assigned_agent}{ext}")

    # --- Wrapper (with JSON repair) ---
    print("\n  [Wrapper] Environment synthesis...")
    wrapper = WrapperAgent(model_name=model_name)
    t0 = time.time()

    # Use manual call + repair to handle JSON failures
    prompt = wrapper._build_prompt(plan)
    raw = llm_call(prompt)
    data = extract_json(raw)

    if not data or not data.get("environments"):
        print(f"  ⚠ Wrapper JSON parse issue — raw output ({len(raw)} chars):")
        print(f"      {raw[:300]}...")
        # Fallback: use rule-based synthesis
        print(f"  ⟳ Falling back to rule-based environment synthesis...")
        wrapper_out = wrapper.synthesize_rule_based(plan)
    else:
        wrapper_out = wrapper._parse_output(json.dumps(data), plan)

    t_wrap = time.time() - t0
    n_envs = len(wrapper_out.environments)
    print(f"  ✓ {n_envs} environments ({t_wrap:.1f}s)")
    for ei in wrapper_out.environments:
        cp_val = ei.checkpoint.expected_value if ei.checkpoint else "?"
        print(f"      [{ei.agent_name}] cp='{cp_val[:60]}'")

    # --- Assemble ---
    instance = BenchmarkInstance(
        seed_data=seed,
        gold_agent_sequence=[m.assigned_agent for m in plan.task_marks],
        gold_checkpoints=[ei.checkpoint for ei in wrapper_out.environments if ei.checkpoint],
        environments=wrapper_out.environments,
        exception_scenarios=wrapper_out.exception_scenarios,
        gold_recovery_plans=wrapper_out.gold_recovery_plans,
    )

    # --- Validation ---
    if not skip_validation and n_envs > 0:
        print("\n  [Validation] Three-tier quality control...")
        validator = ValidationCommittee()
        validator._llm = val_llm_call
        t0 = time.time()
        report = validator.validate(instance)
        status = "PASSED" if report.passed else "FAILED"
        print(f"  ✓ Validation {status} ({time.time() - t0:.1f}s)")
        for r in report.results:
            print(f"      {r.tier.value}: {'PASS' if r.passed else 'FAIL'} — {r.reasoning[:100]}")
    elif n_envs == 0:
        print("\n  [Validation] SKIPPED — no environments generated")

    print(f"\n  ⏱ Generation: {time.time() - t_start:.1f}s")
    return instance


def save_benchmark(instance: BenchmarkInstance, index: int) -> str:
    """Save a benchmark instance to bench/."""
    os.makedirs(BENCH_DIR, exist_ok=True)
    data = {
        "index": index,
        "generated_at": datetime.now().isoformat(),
        "phase": "generation",
        "instance": serialize_instance(instance),
    }
    filename = f"{index:03d}_{instance.seed_data.id}.json"
    filepath = os.path.join(BENCH_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved: bench/{filename}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="IWG Phase 1: Benchmark Generation")
    parser.add_argument("--seed-id", type=str, help="Generate a single seed by id")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated seeds")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tier")
    args = parser.parse_args()

    print("=" * 70)
    print("IWG Phase 1: Benchmark Generation (Scout → Wrapper → Validation)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    config = load_config()
    seeds_data = load_seeds()
    all_seeds = seeds_data["seeds"]

    if args.seed_id:
        all_seeds = [s for s in all_seeds if s["id"] == args.seed_id]
        if not all_seeds:
            print(f"ERROR: seed '{args.seed_id}' not found")
            sys.exit(1)

    total = len(all_seeds)
    print(f"\nSeeds: {total} | Model: {config['openai']['model']}")
    print(f"Output: {BENCH_DIR}/ | Resume: {args.resume}")

    llm_call = create_llm_callable(config)

    # Validation committee needs (model_name, prompt) -> response
    def val_llm_call(model_name: str, prompt: str) -> str:
        return llm_call(prompt)

    results = []
    for i, seed_dict in enumerate(all_seeds):
        idx = i + 1

        # Resume check: look for existing file with this seed_id
        if args.resume:
            existing = [
                f for f in os.listdir(BENCH_DIR)
                if f.endswith(".json") and seed_dict["id"] in f
            ] if os.path.isdir(BENCH_DIR) else []
            if existing:
                print(f"\n  ⏭ {seed_dict['id']}: already exists ({existing[0]}), skipping.")
                results.append({"seed_id": seed_dict["id"], "status": "skipped"})
                continue

        try:
            instance = generate_one(seed_dict, llm_call, val_llm_call, idx, total,
                                    args.skip_validation, config["openai"]["model"])
            if instance is None:
                results.append({"seed_id": seed_dict["id"], "status": "failed", "error": "None returned"})
                continue

            filepath = save_benchmark(instance, idx)
            results.append({
                "seed_id": seed_dict["id"],
                "status": "success",
                "instance_id": instance.id,
                "file": os.path.basename(filepath),
                "num_environments": len(instance.environments),
                "num_checkpoints": len(instance.gold_checkpoints),
                "validation_passed": instance.validation_report.passed if instance.validation_report else None,
            })
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            traceback.print_exc()
            results.append({"seed_id": seed_dict["id"], "status": "error", "error": str(e)})

    # Summary
    print("\n\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    success = sum(1 for r in results if r["status"] == "success")
    print(f"  Total: {total} | Success: {success} | Errors: {sum(1 for r in results if r['status'] == 'error')}")
    for r in results:
        icon = "✓" if r["status"] == "success" else ("⏭" if r["status"] == "skipped" else "❌")
        envs = r.get("num_environments", "?")
        val = "PASS" if r.get("validation_passed") else ("FAIL" if r.get("validation_passed") is False else "N/A")
        print(f"  {icon} {r['seed_id']}: envs={envs} val={val} → {r.get('file', 'N/A')}")


if __name__ == "__main__":
    main()
