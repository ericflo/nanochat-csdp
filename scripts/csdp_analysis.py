"""
CSDP Comparative Analysis Script.

Analyzes results across all curriculum conditions and generates publication-quality
reports, tables, and figures.

Usage:
    python -m scripts.csdp_analysis --runs_dir=/path/to/experiment
    python -m scripts.csdp_analysis --runs_dir=/path/to/experiment --compare
    python -m scripts.csdp_analysis --runs_dir=/path/to/experiment --latex
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RunMetrics:
    """Metrics from a single curriculum run."""
    curriculum: str
    # Standard benchmarks
    mmlu: float = 0.0
    arc_easy: float = 0.0
    arc_challenge: float = 0.0
    gsm8k: float = 0.0
    humaneval: float = 0.0
    spellingbee: float = 0.0
    # CSDP-specific metrics
    self_knowledge: float = 0.0
    calibration: float = 0.0
    consistency: float = 0.0
    ood_self_knowledge: float = 0.0
    social_engineering: float = 0.0
    tone_leakage: float = 0.0
    csdp_score: float = 0.0
    # Training metrics
    val_loss: float = 0.0
    train_time_minutes: float = 0.0
    # Raw data
    raw_data: Dict[str, Any] = field(default_factory=dict)


def load_run_metrics(run_dir: str, curriculum: str) -> Optional[RunMetrics]:
    """Load metrics from a single run directory."""
    metrics = RunMetrics(curriculum=curriculum)

    # Try to load from various possible locations
    report_path = os.path.join(run_dir, "report", "report.md")
    summary_path = os.path.join(run_dir, "run_summary.json")
    csdp_logs_path = os.path.join(run_dir, "csdp_runs")

    # Load from report sections if available
    sections_dir = os.path.join(run_dir, "report", "sections")
    if os.path.exists(sections_dir):
        for fname in os.listdir(sections_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(sections_dir, fname), 'r') as f:
                        data = json.load(f)
                        _extract_metrics_from_section(metrics, data, fname)
                except Exception as e:
                    print(f"Warning: Could not load {fname}: {e}")

    # Load CSDP-specific evaluation results
    csdp_eval_path = os.path.join(run_dir, "csdp_eval_results.json")
    if os.path.exists(csdp_eval_path):
        try:
            with open(csdp_eval_path, 'r') as f:
                csdp_data = json.load(f)
                metrics.self_knowledge = csdp_data.get('SelfKnowledge', 0.0)
                metrics.calibration = csdp_data.get('Calibration', 0.0)
                metrics.consistency = csdp_data.get('Consistency', 0.0)
                metrics.ood_self_knowledge = csdp_data.get('OODSelfKnowledge', 0.0)
                metrics.social_engineering = csdp_data.get('SocialEngineering', 0.0)
                metrics.tone_leakage = csdp_data.get('ToneLeakage', 0.0)
                metrics.csdp_score = csdp_data.get('CSDP_Score', 0.0)
        except Exception as e:
            print(f"Warning: Could not load CSDP eval results: {e}")

    return metrics


def _extract_metrics_from_section(metrics: RunMetrics, data: Any, filename: str):
    """Extract metrics from a report section."""
    if isinstance(data, list):
        for item in data:
            _extract_metrics_from_section(metrics, item, filename)
    elif isinstance(data, dict):
        # Look for known metric keys
        for key, value in data.items():
            if isinstance(value, (int, float)):
                key_lower = key.lower().replace(' ', '_').replace('-', '_')
                if 'mmlu' in key_lower:
                    metrics.mmlu = max(metrics.mmlu, value)
                elif 'arc_easy' in key_lower or 'arceasy' in key_lower:
                    metrics.arc_easy = max(metrics.arc_easy, value)
                elif 'arc_challenge' in key_lower or 'arcchallenge' in key_lower:
                    metrics.arc_challenge = max(metrics.arc_challenge, value)
                elif 'gsm8k' in key_lower:
                    metrics.gsm8k = max(metrics.gsm8k, value)
                elif 'humaneval' in key_lower:
                    metrics.humaneval = max(metrics.humaneval, value)
                elif 'spellingbee' in key_lower:
                    metrics.spellingbee = max(metrics.spellingbee, value)
                elif 'val_loss' in key_lower or 'validation_loss' in key_lower:
                    metrics.val_loss = value
                elif 'self_knowledge' in key_lower or 'selfknowledge' in key_lower:
                    metrics.self_knowledge = value
                elif 'calibration' in key_lower:
                    metrics.calibration = value
                elif 'consistency' in key_lower:
                    metrics.consistency = value

        metrics.raw_data.update(data)


def load_all_runs(runs_dir: str) -> Dict[str, RunMetrics]:
    """Load metrics from all curriculum runs."""
    runs = {}
    curricula = ['none', 'aria', 'sage', 'nova', 'heart', 'bare']

    for curriculum in curricula:
        # Check for curriculum-specific directory
        curriculum_dir = os.path.join(runs_dir, curriculum)
        if os.path.isdir(curriculum_dir):
            metrics = load_run_metrics(curriculum_dir, curriculum)
            if metrics:
                runs[curriculum] = metrics
            else:
                print(f"Warning: Could not load metrics for {curriculum}")
        else:
            # Check for report file directly
            report_path = os.path.join(runs_dir, f"report_{curriculum}.md")
            if os.path.exists(report_path):
                # Create minimal metrics from report
                runs[curriculum] = RunMetrics(curriculum=curriculum)

    return runs


def generate_comparison_table(runs: Dict[str, RunMetrics]) -> str:
    """Generate a markdown comparison table."""
    if not runs:
        return "No runs found."

    lines = []
    lines.append("# CSDP Experiment Results Comparison\n")

    # Standard benchmarks table
    lines.append("## Standard Benchmarks\n")
    lines.append("| Curriculum | MMLU | ARC-Easy | ARC-Chal | GSM8K | HumanEval | SpellingBee |")
    lines.append("|------------|------|----------|----------|-------|-----------|-------------|")

    for curriculum in ['none', 'aria', 'sage', 'nova', 'heart', 'bare']:
        if curriculum in runs:
            m = runs[curriculum]
            lines.append(f"| {curriculum:10} | {m.mmlu:.3f} | {m.arc_easy:.3f} | {m.arc_challenge:.3f} | {m.gsm8k:.3f} | {m.humaneval:.3f} | {m.spellingbee:.3f} |")

    lines.append("")

    # CSDP-specific metrics table
    lines.append("## CSDP-Specific Metrics\n")
    lines.append("| Curriculum | SelfKnow | Calibr | Consist | OOD-SK | SocEng | ToneLeak | CSDP Score |")
    lines.append("|------------|----------|--------|---------|--------|--------|----------|------------|")

    for curriculum in ['none', 'aria', 'sage', 'nova', 'heart', 'bare']:
        if curriculum in runs:
            m = runs[curriculum]
            lines.append(f"| {curriculum:10} | {m.self_knowledge:.3f} | {m.calibration:.3f} | {m.consistency:.3f} | {m.ood_self_knowledge:.3f} | {m.social_engineering:.3f} | {m.tone_leakage:.3f} | {m.csdp_score:.3f} |")

    lines.append("")

    return "\n".join(lines)


def generate_latex_table(runs: Dict[str, RunMetrics]) -> str:
    """Generate LaTeX tables for publication."""
    if not runs:
        return "% No runs found"

    lines = []
    lines.append("% CSDP Experiment Results - Standard Benchmarks")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Standard Benchmark Performance by Curriculum}")
    lines.append("\\label{tab:standard-benchmarks}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Curriculum & MMLU & ARC-E & ARC-C & GSM8K & HumanEval & SpellingBee \\\\")
    lines.append("\\midrule")

    for curriculum in ['none', 'aria', 'sage', 'nova', 'heart', 'bare']:
        if curriculum in runs:
            m = runs[curriculum]
            name = "Baseline" if curriculum == "none" else curriculum.upper()
            lines.append(f"{name} & {m.mmlu:.1%} & {m.arc_easy:.1%} & {m.arc_challenge:.1%} & {m.gsm8k:.1%} & {m.humaneval:.1%} & {m.spellingbee:.1%} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # CSDP metrics table
    lines.append("% CSDP-Specific Metrics")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{CSDP-Specific Evaluation Metrics by Curriculum}")
    lines.append("\\label{tab:csdp-metrics}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append("Curriculum & Self-Know & Calibr. & Consist. & OOD-SK & Soc.Eng & Tone & CSDP \\\\")
    lines.append("\\midrule")

    for curriculum in ['none', 'aria', 'sage', 'nova', 'heart', 'bare']:
        if curriculum in runs:
            m = runs[curriculum]
            name = "Baseline" if curriculum == "none" else curriculum.upper()
            lines.append(f"{name} & {m.self_knowledge:.1%} & {m.calibration:.1%} & {m.consistency:.1%} & {m.ood_self_knowledge:.1%} & {m.social_engineering:.1%} & {m.tone_leakage:.1%} & {m.csdp_score:.1%} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def compute_deltas(runs: Dict[str, RunMetrics]) -> Dict[str, Dict[str, float]]:
    """Compute performance deltas relative to baseline."""
    if 'none' not in runs:
        return {}

    baseline = runs['none']
    deltas = {}

    for curriculum, metrics in runs.items():
        if curriculum == 'none':
            continue

        deltas[curriculum] = {
            'mmlu': metrics.mmlu - baseline.mmlu,
            'arc_easy': metrics.arc_easy - baseline.arc_easy,
            'arc_challenge': metrics.arc_challenge - baseline.arc_challenge,
            'gsm8k': metrics.gsm8k - baseline.gsm8k,
            'humaneval': metrics.humaneval - baseline.humaneval,
            'self_knowledge': metrics.self_knowledge - baseline.self_knowledge,
            'calibration': metrics.calibration - baseline.calibration,
            'consistency': metrics.consistency - baseline.consistency,
            'csdp_score': metrics.csdp_score - baseline.csdp_score,
        }

    return deltas


def generate_delta_analysis(runs: Dict[str, RunMetrics]) -> str:
    """Generate analysis of deltas from baseline."""
    deltas = compute_deltas(runs)
    if not deltas:
        return "No baseline found for delta analysis."

    lines = []
    lines.append("## Performance Delta from Baseline\n")
    lines.append("Positive values indicate improvement over baseline (no CSDP).\n")

    lines.append("| Curriculum | MMLU | ARC-E | GSM8K | Self-Know | Calibr | CSDP Score |")
    lines.append("|------------|------|-------|-------|-----------|--------|------------|")

    for curriculum in ['aria', 'sage', 'nova', 'heart', 'bare']:
        if curriculum in deltas:
            d = deltas[curriculum]
            lines.append(f"| {curriculum:10} | {d['mmlu']:+.3f} | {d['arc_easy']:+.3f} | {d['gsm8k']:+.3f} | {d['self_knowledge']:+.3f} | {d['calibration']:+.3f} | {d['csdp_score']:+.3f} |")

    lines.append("")

    # Find best curriculum for each metric
    lines.append("### Best Curriculum by Metric\n")
    metrics_to_check = ['mmlu', 'arc_easy', 'gsm8k', 'self_knowledge', 'calibration', 'csdp_score']

    for metric in metrics_to_check:
        best_curriculum = max(deltas.keys(), key=lambda c: deltas[c][metric])
        best_delta = deltas[best_curriculum][metric]
        lines.append(f"- **{metric}**: {best_curriculum} ({best_delta:+.3f})")

    return "\n".join(lines)


def generate_summary_analysis(runs: Dict[str, RunMetrics]) -> str:
    """Generate a summary analysis of the results."""
    if not runs:
        return "No runs to analyze."

    lines = []
    lines.append("# CSDP Experiment Analysis Summary\n")

    # Overview
    lines.append("## Overview\n")
    lines.append(f"- **Number of curricula tested**: {len(runs)}")
    lines.append(f"- **Curricula**: {', '.join(runs.keys())}")
    lines.append("")

    # Key findings
    if len(runs) > 1:
        lines.append("## Key Findings\n")

        # Best overall CSDP score
        best_csdp = max(runs.items(), key=lambda x: x[1].csdp_score)
        lines.append(f"- **Best CSDP Score**: {best_csdp[0]} ({best_csdp[1].csdp_score:.3f})")

        # Best standard benchmark performance
        def avg_std_score(m):
            return (m.mmlu + m.arc_easy + m.arc_challenge + m.gsm8k + m.humaneval) / 5

        best_std = max(runs.items(), key=lambda x: avg_std_score(x[1]))
        lines.append(f"- **Best Standard Benchmarks**: {best_std[0]} (avg: {avg_std_score(best_std[1]):.3f})")

        # Best self-knowledge
        best_sk = max(runs.items(), key=lambda x: x[1].self_knowledge)
        lines.append(f"- **Best Self-Knowledge**: {best_sk[0]} ({best_sk[1].self_knowledge:.3f})")

        # Best calibration
        best_cal = max(runs.items(), key=lambda x: x[1].calibration)
        lines.append(f"- **Best Calibration**: {best_cal[0]} ({best_cal[1].calibration:.3f})")

        lines.append("")

    # Add comparison table
    lines.append(generate_comparison_table(runs))

    # Add delta analysis
    lines.append(generate_delta_analysis(runs))

    return "\n".join(lines)


def save_analysis(runs: Dict[str, RunMetrics], output_dir: str):
    """Save all analysis outputs to directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save markdown summary
    summary = generate_summary_analysis(runs)
    with open(os.path.join(output_dir, "summary.md"), 'w') as f:
        f.write(summary)

    # Save LaTeX tables
    latex = generate_latex_table(runs)
    with open(os.path.join(output_dir, "tables.tex"), 'w') as f:
        f.write(latex)

    # Save raw metrics as JSON
    metrics_dict = {}
    for curriculum, m in runs.items():
        metrics_dict[curriculum] = {
            'mmlu': m.mmlu,
            'arc_easy': m.arc_easy,
            'arc_challenge': m.arc_challenge,
            'gsm8k': m.gsm8k,
            'humaneval': m.humaneval,
            'spellingbee': m.spellingbee,
            'self_knowledge': m.self_knowledge,
            'calibration': m.calibration,
            'consistency': m.consistency,
            'ood_self_knowledge': m.ood_self_knowledge,
            'social_engineering': m.social_engineering,
            'tone_leakage': m.tone_leakage,
            'csdp_score': m.csdp_score,
        }

    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Save deltas
    deltas = compute_deltas(runs)
    with open(os.path.join(output_dir, "deltas.json"), 'w') as f:
        json.dump(deltas, f, indent=2)

    print(f"Analysis saved to {output_dir}")
    print(f"  - summary.md: Markdown summary report")
    print(f"  - tables.tex: LaTeX tables for publication")
    print(f"  - metrics.json: Raw metrics data")
    print(f"  - deltas.json: Delta from baseline")


def main():
    parser = argparse.ArgumentParser(description="CSDP Comparative Analysis")
    parser.add_argument('--runs_dir', type=str, required=True,
                        help="Directory containing curriculum runs")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory for analysis (default: runs_dir/analysis)")
    parser.add_argument('--compare', action='store_true',
                        help="Print comparison table to stdout")
    parser.add_argument('--latex', action='store_true',
                        help="Print LaTeX tables to stdout")
    parser.add_argument('--deltas', action='store_true',
                        help="Print delta analysis to stdout")
    args = parser.parse_args()

    # Load runs
    runs = load_all_runs(args.runs_dir)

    if not runs:
        print(f"No runs found in {args.runs_dir}")
        print("Expected structure:")
        print("  runs_dir/")
        print("    none/     (or report_none.md)")
        print("    aria/     (or report_aria.md)")
        print("    sage/     ...")
        return

    print(f"Loaded {len(runs)} curriculum runs: {list(runs.keys())}")

    # Handle output modes
    if args.compare:
        print(generate_comparison_table(runs))
    elif args.latex:
        print(generate_latex_table(runs))
    elif args.deltas:
        print(generate_delta_analysis(runs))
    else:
        # Full analysis
        output_dir = args.output_dir or os.path.join(args.runs_dir, "analysis")
        save_analysis(runs, output_dir)

        # Print summary to stdout as well
        print("\n" + generate_summary_analysis(runs))


if __name__ == "__main__":
    main()
