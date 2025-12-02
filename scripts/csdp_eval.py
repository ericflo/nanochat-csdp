"""
Run comprehensive CSDP evaluations on a trained model.

This script runs all CSDP-specific evaluation tasks and computes
aggregate metrics for comparing curriculum effectiveness.

Usage:
    python -m scripts.csdp_eval -i sft --curriculum heart
    python -m scripts.csdp_eval -i mid --curriculum aria --max_problems 100

Or with torchrun:
    torchrun --standalone --nproc_per_node=8 -m scripts.csdp_eval -- -i sft --curriculum heart
"""

import os
import argparse
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.report import get_report

from tasks.csdp_metrics import (
    SelfKnowledgeTask,
    CalibrationTask,
    ConsistencyTask,
    OODSelfKnowledgeTask,
    SocialEngineeringTask,
    ToneLeakageTask,
    compute_csdp_score,
)


# Task registry
CSDP_TASKS = {
    'SelfKnowledge': SelfKnowledgeTask,
    'Calibration': CalibrationTask,
    'Consistency': ConsistencyTask,
    'OODSelfKnowledge': OODSelfKnowledgeTask,
    'SocialEngineering': SocialEngineeringTask,
    'ToneLeakage': ToneLeakageTask,
}


def run_generative_eval(task, tokenizer, model, engine, device,
                        num_samples=1, max_tokens=256, temperature=0.0,
                        max_problems=-1):
    """
    Run generative evaluation on a CSDP task.

    Args:
        task: Task instance
        tokenizer: Tokenizer
        model: Model
        engine: Engine for generation
        device: Device
        num_samples: Number of samples per prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        max_problems: Max problems to evaluate (-1 = all)

    Returns:
        Accuracy (float)
    """
    ddp = dist.is_initialized()
    ddp_rank = dist.get_rank() if ddp else 0
    ddp_world_size = dist.get_world_size() if ddp else 1

    num_problems = task.num_examples()
    if max_problems > 0:
        num_problems = min(num_problems, max_problems)

    correct = 0
    total = 0

    for i in range(ddp_rank, num_problems, ddp_world_size):
        example = task.get_example(i)
        conversation = example

        # Encode prompt
        prompt_tokens = tokenizer.render_for_completion(conversation)

        # Generate
        results, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Decode and evaluate
        prefix_len = len(prompt_tokens)
        completions = [tokenizer.decode(r[prefix_len:]) for r in results]

        # Take best of samples
        outcomes = [task.evaluate(conversation, c) for c in completions]
        best_outcome = max(outcomes) if outcomes else 0.0

        correct += best_outcome
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
    return accuracy


def run_csdp_eval(model, tokenizer, engine, device, curriculum,
                  tasks=None, max_problems=-1):
    """
    Run all CSDP evaluations.

    Args:
        model: Model
        tokenizer: Tokenizer
        engine: Engine
        device: Device
        curriculum: Curriculum name (for logging)
        tasks: List of task names to run (None = all)
        max_problems: Max problems per task

    Returns:
        Dict of task_name -> accuracy
    """
    if tasks is None:
        tasks = list(CSDP_TASKS.keys())

    results = {}

    for task_name in tasks:
        if task_name not in CSDP_TASKS:
            print0(f"Unknown task: {task_name}, skipping")
            continue

        task_class = CSDP_TASKS[task_name]
        task = task_class()

        print0(f"Running {task_name}...")

        accuracy = run_generative_eval(
            task, tokenizer, model, engine, device,
            max_problems=max_problems,
        )

        results[task_name] = accuracy
        print0(f"  {task_name}: {accuracy:.4f}")

    # Compute aggregate score
    csdp_score = compute_csdp_score(results)
    results['CSDP_Score'] = csdp_score
    print0(f"CSDP Score: {csdp_score:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run CSDP evaluations")
    parser.add_argument('-i', '--input', type=str, required=True,
                        choices=['base', 'mid', 'sft'],
                        help="Model checkpoint to evaluate")
    parser.add_argument('--curriculum', type=str, default="none",
                        help="Curriculum used for training (for logging)")
    parser.add_argument('--model_tag', type=str, default=None,
                        help="Model tag")
    parser.add_argument('--step', type=int, default=None,
                        help="Checkpoint step")
    parser.add_argument('--tasks', type=str, default=None,
                        help="Comma-separated list of tasks to run")
    parser.add_argument('--max_problems', type=int, default=-1,
                        help="Max problems per task")
    parser.add_argument('--device_type', type=str, default="",
                        help="Device type (cuda|cpu|mps)")
    args = parser.parse_args()

    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model
    print0(f"Loading model from {args.input}...")
    model, tokenizer, meta = load_model(args.input, device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Parse task list
    tasks = None
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]

    # Run evaluations
    print0(f"\nRunning CSDP evaluations (curriculum: {args.curriculum})")
    print0("=" * 50)

    model.eval()
    with torch.no_grad(), autocast_ctx:
        results = run_csdp_eval(
            model, tokenizer, engine, device,
            curriculum=args.curriculum,
            tasks=tasks,
            max_problems=args.max_problems,
        )

    # Log results
    if ddp_rank == 0:
        print0("\n" + "=" * 50)
        print0("CSDP Evaluation Results:")
        for task_name, accuracy in sorted(results.items()):
            print0(f"  {task_name}: {accuracy:.4f}")

        # Log to report
        get_report().log(
            section=f"CSDP Evaluation ({args.input})",
            data=[
                {"curriculum": args.curriculum},
                results,
            ]
        )

    compute_cleanup()


if __name__ == "__main__":
    main()
