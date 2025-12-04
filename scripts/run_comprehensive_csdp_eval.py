"""
Comprehensive CSDP Evaluation Script

Runs extended OOD probes and all CSDP evaluations across:
- All 5 curricula (aria, sage, nova, heart, bare)
- All 3 stages (base, mid, sft)
- Multiple runs for variance estimation

This generates the data needed for publication-quality analysis.

Usage:
    # Full evaluation (all curricula, all stages)
    python scripts/run_comprehensive_csdp_eval.py

    # Single curriculum
    python scripts/run_comprehensive_csdp_eval.py --curriculum aria

    # Specific stage
    python scripts/run_comprehensive_csdp_eval.py --stage sft

    # Variance estimation (5 runs)
    python scripts/run_comprehensive_csdp_eval.py --n_runs 5 --temperature 0.3
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.engine import Engine

from tasks.csdp_metrics import (
    SelfKnowledgeTask,
    CalibrationTask,
    ConsistencyTask,
    OODSelfKnowledgeTask,
    SocialEngineeringTask,
    ToneLeakageTask,
    compute_csdp_score,
)
from tasks.extended_ood_probes import ExtendedOODSelfKnowledgeTask


CURRICULA = ["aria", "sage", "nova", "heart", "bare"]
STAGES = ["base", "mid", "sft"]
RESULTS_DIR = Path(__file__).parent.parent / "paper" / "data"


def run_eval_task(task, tokenizer, model, engine, device,
                  max_tokens=256, temperature=0.0, max_problems=-1):
    """Run evaluation on a single task."""
    ddp = dist.is_initialized()
    ddp_rank = dist.get_rank() if ddp else 0
    ddp_world_size = dist.get_world_size() if ddp else 1

    num_problems = task.num_examples()
    if max_problems > 0:
        num_problems = min(num_problems, max_problems)

    correct = 0
    total = 0
    per_example_results = []

    for i in range(ddp_rank, num_problems, ddp_world_size):
        example = task.get_example(i)

        # Encode prompt
        prompt_tokens = tokenizer.render_for_completion(example)

        # Generate
        results, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Decode
        prefix_len = len(prompt_tokens)
        response = tokenizer.decode(results[0][prefix_len:])

        # Evaluate
        score = task.evaluate(example, response)

        per_example_results.append({
            "index": i,
            "prompt": example["messages"][0]["content"],
            "response": response,
            "score": score,
            "category": example.get("category", "default"),
        })

        correct += score
        total += 1

    # Aggregate across ranks
    if ddp:
        correct_tensor = torch.tensor(correct, device=device, dtype=torch.float32)
        total_tensor = torch.tensor(total, device=device, dtype=torch.float32)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_example": per_example_results,
    }


def run_extended_ood_eval(tokenizer, model, engine, device,
                          temperature=0.0, max_problems=-1):
    """Run extended OOD self-knowledge evaluation with category breakdown."""
    task = ExtendedOODSelfKnowledgeTask()
    results = run_eval_task(task, tokenizer, model, engine, device,
                           temperature=temperature, max_problems=max_problems)

    # Compute per-category scores
    category_scores = {}
    for ex in results["per_example"]:
        cat = ex["category"]
        if cat not in category_scores:
            category_scores[cat] = {"correct": 0, "total": 0}
        category_scores[cat]["correct"] += ex["score"]
        category_scores[cat]["total"] += 1

    category_accuracy = {
        cat: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for cat, data in category_scores.items()
    }

    results["category_scores"] = category_accuracy
    results["category_details"] = category_scores

    return results


def run_all_csdp_evals(tokenizer, model, engine, device,
                       temperature=0.0, max_problems=-1):
    """Run all CSDP evaluation tasks."""
    tasks = {
        'SelfKnowledge': SelfKnowledgeTask(),
        'Calibration': CalibrationTask(),
        'OODSelfKnowledge': OODSelfKnowledgeTask(),
        'SocialEngineering': SocialEngineeringTask(),
        'ToneLeakage': ToneLeakageTask(),
        'ExtendedOOD': ExtendedOODSelfKnowledgeTask(),
    }

    results = {}
    for name, task in tasks.items():
        print0(f"  Running {name}...")

        if name == "ExtendedOOD":
            task_results = run_extended_ood_eval(
                tokenizer, model, engine, device,
                temperature=temperature, max_problems=max_problems
            )
        else:
            task_results = run_eval_task(
                task, tokenizer, model, engine, device,
                temperature=temperature, max_problems=max_problems
            )

        results[name] = task_results
        print0(f"    {name}: {task_results['accuracy']:.4f}")

    # Compute aggregate CSDP score
    simple_results = {k: v["accuracy"] for k, v in results.items()}
    csdp_score = compute_csdp_score(simple_results)
    results["CSDP_Score"] = csdp_score

    return results


def evaluate_curriculum_stage(curriculum: str, stage: str,
                              temperature: float = 0.0,
                              max_problems: int = -1,
                              device_type: str = "",
                              exp_dir: Path = None):
    """Evaluate a specific curriculum at a specific stage."""
    print0(f"\n{'='*60}")
    print0(f"Evaluating: {curriculum.upper()} @ {stage.upper()}")
    print0(f"{'='*60}")

    # Setup
    if device_type == "":
        device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda" else nullcontext()
    )

    # Determine checkpoint path
    if exp_dir is None:
        exp_dir = Path(__file__).parent.parent / "experimental_data_and_results"
    curriculum_dir = Path(exp_dir) / curriculum

    if stage == "base":
        checkpoint_dir = curriculum_dir / "base_checkpoints"
    elif stage == "mid":
        checkpoint_dir = curriculum_dir / "mid_checkpoints"
    elif stage == "sft":
        checkpoint_dir = curriculum_dir / "chatsft_checkpoints"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if not checkpoint_dir.exists():
        print0(f"Warning: Checkpoint not found at {checkpoint_dir}")
        compute_cleanup()
        return None

    # Check for depth subdirectory (e.g., d20, d32) - nanochat stores checkpoints there
    depth_dirs = list(checkpoint_dir.glob("d*"))
    if depth_dirs:
        checkpoint_dir = depth_dirs[0]  # Use first depth dir found
        print0(f"Found depth subdirectory: {checkpoint_dir.name}")

    # Load model
    print0(f"Loading model from {checkpoint_dir}...")
    try:
        model, tokenizer, meta = load_model_from_dir(str(checkpoint_dir), device, phase="eval")
        engine = Engine(model, tokenizer)
    except Exception as e:
        import traceback
        print0(f"Error loading model: {e}")
        print0(f"Traceback: {traceback.format_exc()}")
        compute_cleanup()
        return None

    # Run evaluations
    model.eval()
    with torch.no_grad(), autocast_ctx:
        results = run_all_csdp_evals(
            tokenizer, model, engine, device,
            temperature=temperature,
            max_problems=max_problems
        )

    results["curriculum"] = curriculum
    results["stage"] = stage
    results["temperature"] = temperature
    results["timestamp"] = datetime.now().isoformat()

    compute_cleanup()
    return results


def save_results(all_results: dict, output_dir: Path):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comprehensive_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print0(f"\nResults saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CSDP Evaluation")
    parser.add_argument("--curriculum", type=str, default="all",
                       help="Curriculum to evaluate (or 'all')")
    parser.add_argument("--stage", type=str, default="all",
                       help="Stage to evaluate (base/mid/sft or 'all')")
    parser.add_argument("--n_runs", type=int, default=1,
                       help="Number of runs for variance estimation")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 for deterministic)")
    parser.add_argument("--max_problems", type=int, default=-1,
                       help="Max problems per task (-1 = all)")
    parser.add_argument("--device_type", type=str, default="",
                       help="Device type (cuda/cpu/mps)")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR),
                       help="Output directory for results")
    parser.add_argument("--exp_dir", type=str, default="",
                       help="Experiment directory containing curriculum folders")
    args = parser.parse_args()

    curricula = CURRICULA if args.curriculum == "all" else [args.curriculum]
    stages = STAGES if args.stage == "all" else [args.stage]
    exp_dir = Path(args.exp_dir) if args.exp_dir else None

    print0(f"\n{'='*60}")
    print0("COMPREHENSIVE CSDP EVALUATION")
    print0(f"{'='*60}")
    print0(f"Curricula: {curricula}")
    print0(f"Stages: {stages}")
    print0(f"Runs per config: {args.n_runs}")
    print0(f"Temperature: {args.temperature}")
    print0(f"{'='*60}\n")

    all_results = {
        "meta": {
            "curricula": curricula,
            "stages": stages,
            "n_runs": args.n_runs,
            "temperature": args.temperature,
            "timestamp": datetime.now().isoformat(),
        },
        "results": []
    }

    for curriculum in curricula:
        for stage in stages:
            for run_idx in range(args.n_runs):
                print0(f"\n--- Run {run_idx + 1}/{args.n_runs} ---")

                results = evaluate_curriculum_stage(
                    curriculum=curriculum,
                    stage=stage,
                    temperature=args.temperature,
                    max_problems=args.max_problems,
                    device_type=args.device_type,
                    exp_dir=exp_dir,
                )

                if results is not None:
                    results["run_idx"] = run_idx
                    all_results["results"].append(results)

    # Save results
    output_file = save_results(all_results, Path(args.output_dir))

    # Print summary
    print0(f"\n{'='*60}")
    print0("SUMMARY")
    print0(f"{'='*60}")

    for curriculum in curricula:
        print0(f"\n{curriculum.upper()}:")
        for stage in stages:
            stage_results = [
                r for r in all_results["results"]
                if r["curriculum"] == curriculum and r["stage"] == stage
            ]
            if stage_results:
                avg_csdp = sum(r["CSDP_Score"] for r in stage_results) / len(stage_results)
                avg_ext_ood = sum(
                    r["ExtendedOOD"]["accuracy"] for r in stage_results
                ) / len(stage_results)
                print0(f"  {stage}: CSDP={avg_csdp:.4f}, ExtendedOOD={avg_ext_ood:.4f}")

    print0(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
