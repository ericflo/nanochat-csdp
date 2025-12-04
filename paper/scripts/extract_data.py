#!/usr/bin/env python3
"""
Data extraction script for the CSDP paper.
Extracts metrics from report markdown files and processes for figure generation.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experimental_data_and_results"
OUTPUT_DIR = Path(__file__).parent

CURRICULA = ["aria", "sage", "nova", "heart", "bare", "none"]
STAGES = ["base", "mid", "sft"]

# Colorblind-friendly palette for curricula
CURRICULUM_COLORS = {
    'aria': '#377eb8',    # Blue - technical
    'sage': '#4daf4a',    # Green - supportive
    'nova': '#984ea3',    # Purple - philosophical
    'heart': '#e41a1c',   # Red - loving
    'bare': '#ff7f00',    # Orange - minimal
    'none': '#999999',    # Gray - no CSDP baseline
}


def parse_report(report_path: Path) -> Dict[str, Any]:
    """Parse a report markdown file and extract all metrics."""
    with open(report_path, 'r') as f:
        content = f.read()

    data = {
        'curriculum': report_path.stem.replace('report_', ''),
        'raw_content': content,
    }

    # Extract sections
    sections = re.split(r'\n## ', content)

    for section in sections:
        if section.startswith('CSDP Configuration'):
            data['csdp_config'] = parse_key_value_section(section)
        elif section.startswith('Base model training'):
            data['base_training'] = parse_key_value_section(section)
        elif section.startswith('Base model loss'):
            data['base_loss'] = parse_key_value_section(section)
        elif section.startswith('Base model evaluation'):
            data['base_eval'] = parse_key_value_section(section)
        elif section.startswith('Midtraining'):
            data['mid_training'] = parse_key_value_section(section)
        elif section.startswith('Chat evaluation mid'):
            data['mid_eval'] = parse_key_value_section(section)
        elif section.startswith('Chat SFT'):
            data['sft_training'] = parse_key_value_section(section)
        elif section.startswith('Chat evaluation sft'):
            data['sft_eval'] = parse_key_value_section(section)
        elif section.startswith('CSDP Evaluation (sft)'):
            data['csdp_eval_sft'] = parse_key_value_section(section)
        elif section.startswith('CSDP Evaluation (mid)'):
            data['csdp_eval_mid'] = parse_key_value_section(section)

    return data


def parse_key_value_section(section: str) -> Dict[str, Any]:
    """Parse a section with key: value pairs."""
    result = {}
    lines = section.split('\n')

    for line in lines:
        # Match "- key: value" format
        match = re.match(r'^-\s+([^:]+):\s+(.+)$', line.strip())
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Try to parse as number
            try:
                if '.' in value:
                    result[key] = float(value.replace(',', ''))
                else:
                    result[key] = int(value.replace(',', ''))
            except ValueError:
                result[key] = value

    return result


def extract_all_curricula() -> Dict[str, Any]:
    """Extract data from all curriculum reports."""
    all_data = {}

    for curriculum in CURRICULA:
        report_path = RESULTS_DIR / f"report_{curriculum}.markdown"
        if report_path.exists():
            print(f"  Processing {curriculum}...")
            all_data[curriculum] = parse_report(report_path)
        else:
            print(f"  Warning: {report_path} not found")

    return all_data


def compute_normalized_metrics(all_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute normalized (0-1) metrics for radar chart."""
    metrics_to_normalize = [
        ('MMLU', 'sft_eval', 0.25, 0.40),  # Expected range
        ('ARC-Easy', 'sft_eval', 0.30, 0.50),
        ('ARC-Challenge', 'sft_eval', 0.25, 0.40),
        ('HumanEval', 'sft_eval', 0.0, 0.15),
        ('ChatCORE', 'sft_eval', 0.20, 0.30),
        ('SelfKnowledge', 'csdp_eval_sft', 0.40, 0.70),
        ('OODSelfKnowledge', 'csdp_eval_sft', 0.10, 0.55),
        ('Calibration', 'csdp_eval_sft', 0.45, 0.60),
    ]

    normalized = {}
    for curriculum, data in all_data.items():
        normalized[curriculum] = {}
        for metric_name, section, min_val, max_val in metrics_to_normalize:
            if section in data and metric_name in data[section]:
                raw_val = data[section][metric_name]
                norm_val = (raw_val - min_val) / (max_val - min_val)
                normalized[curriculum][metric_name] = max(0, min(1, norm_val))
            else:
                normalized[curriculum][metric_name] = 0.0

    return normalized


def extract_training_dynamics(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training dynamics across stages."""
    dynamics = {}

    for curriculum, data in all_data.items():
        dynamics[curriculum] = {
            'base_bpb': data.get('base_training', {}).get('Final validation bpb', None),
            'mid_bpb': data.get('mid_training', {}).get('Minimum validation bpb', None),
            'sft_train_loss': data.get('sft_training', {}).get('Training loss', None),
            'sft_val_loss': data.get('sft_training', {}).get('Validation loss', None),
            'training_time_m': data.get('base_training', {}).get('Total training time', '0m').replace('m', ''),
        }

    return dynamics


def extract_csdp_metrics(all_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract CSDP-specific metrics for all curricula and stages."""
    csdp_metrics = {}

    for curriculum, data in all_data.items():
        csdp_metrics[curriculum] = {
            'mid': {},
            'sft': {},
        }

        if 'csdp_eval_mid' in data:
            csdp_metrics[curriculum]['mid'] = {
                k: v for k, v in data['csdp_eval_mid'].items()
                if isinstance(v, (int, float)) and k != 'curriculum'
            }

        if 'csdp_eval_sft' in data:
            csdp_metrics[curriculum]['sft'] = {
                k: v for k, v in data['csdp_eval_sft'].items()
                if isinstance(v, (int, float)) and k != 'curriculum'
            }

    return csdp_metrics


def extract_benchmark_metrics(all_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract standard benchmark metrics for all curricula."""
    benchmarks = {}

    for curriculum, data in all_data.items():
        benchmarks[curriculum] = {
            'mid': {},
            'sft': {},
            'base': {},
        }

        if 'mid_eval' in data:
            benchmarks[curriculum]['mid'] = {
                k: v for k, v in data['mid_eval'].items()
                if isinstance(v, (int, float))
            }

        if 'sft_eval' in data:
            benchmarks[curriculum]['sft'] = {
                k: v for k, v in data['sft_eval'].items()
                if isinstance(v, (int, float))
            }

        if 'base_eval' in data:
            benchmarks[curriculum]['base'] = {
                k: v for k, v in data['base_eval'].items()
                if isinstance(v, (int, float))
            }

    return benchmarks


def compute_summary_stats(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics across curricula."""
    summary = {
        'best_mmlu': {'curriculum': None, 'value': 0},
        'best_arc_easy': {'curriculum': None, 'value': 0},
        'best_chatcore': {'curriculum': None, 'value': 0},
        'best_self_knowledge': {'curriculum': None, 'value': 0},
        'best_ood_self_knowledge': {'curriculum': None, 'value': 0},
        'best_csdp_score': {'curriculum': None, 'value': 0},
        'lowest_base_bpb': {'curriculum': None, 'value': float('inf')},
        'lowest_sft_loss': {'curriculum': None, 'value': float('inf')},
    }

    for curriculum, data in all_data.items():
        # SFT metrics
        if 'sft_eval' in data:
            sft = data['sft_eval']
            if sft.get('MMLU', 0) > summary['best_mmlu']['value']:
                summary['best_mmlu'] = {'curriculum': curriculum, 'value': sft['MMLU']}
            if sft.get('ARC-Easy', 0) > summary['best_arc_easy']['value']:
                summary['best_arc_easy'] = {'curriculum': curriculum, 'value': sft['ARC-Easy']}
            if sft.get('ChatCORE', 0) > summary['best_chatcore']['value']:
                summary['best_chatcore'] = {'curriculum': curriculum, 'value': sft.get('ChatCORE', 0)}

        # CSDP metrics
        if 'csdp_eval_sft' in data:
            csdp = data['csdp_eval_sft']
            if csdp.get('SelfKnowledge', 0) > summary['best_self_knowledge']['value']:
                summary['best_self_knowledge'] = {'curriculum': curriculum, 'value': csdp['SelfKnowledge']}
            if csdp.get('OODSelfKnowledge', 0) > summary['best_ood_self_knowledge']['value']:
                summary['best_ood_self_knowledge'] = {'curriculum': curriculum, 'value': csdp['OODSelfKnowledge']}
            if csdp.get('CSDP_Score', 0) > summary['best_csdp_score']['value']:
                summary['best_csdp_score'] = {'curriculum': curriculum, 'value': csdp['CSDP_Score']}

        # Training metrics
        if 'base_training' in data:
            base_bpb = data['base_training'].get('Final validation bpb', float('inf'))
            if base_bpb < summary['lowest_base_bpb']['value']:
                summary['lowest_base_bpb'] = {'curriculum': curriculum, 'value': base_bpb}

        if 'sft_training' in data:
            sft_loss = data['sft_training'].get('Training loss', float('inf'))
            if sft_loss < summary['lowest_sft_loss']['value']:
                summary['lowest_sft_loss'] = {'curriculum': curriculum, 'value': sft_loss}

    return summary


def main():
    """Main extraction function."""
    print("Extracting CSDP experiment data...")
    print(f"Looking in: {RESULTS_DIR}")

    # Extract raw data from all reports
    all_data = extract_all_curricula()

    if not all_data:
        print("Error: No data extracted!")
        return None

    print(f"\nProcessing {len(all_data)} curricula...")

    # Compute derived metrics
    extracted = {
        'curricula': list(all_data.keys()),
        'colors': CURRICULUM_COLORS,
        'raw_data': {k: {kk: vv for kk, vv in v.items() if kk != 'raw_content'}
                     for k, v in all_data.items()},
        'normalized_metrics': compute_normalized_metrics(all_data),
        'training_dynamics': extract_training_dynamics(all_data),
        'csdp_metrics': extract_csdp_metrics(all_data),
        'benchmark_metrics': extract_benchmark_metrics(all_data),
        'summary': compute_summary_stats(all_data),
    }

    # Save extracted data
    output_path = OUTPUT_DIR / "extracted_data.json"
    with open(output_path, 'w') as f:
        json.dump(extracted, f, indent=2, default=str)

    print(f"\nData saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Curricula: {extracted['curricula']}")
    print("\nBest performers:")
    for key, val in extracted['summary'].items():
        if val['curriculum']:
            print(f"  {key}: {val['curriculum']} ({val['value']:.4f})")

    return extracted


if __name__ == '__main__':
    main()
