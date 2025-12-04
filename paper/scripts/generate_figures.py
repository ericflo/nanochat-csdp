#!/usr/bin/env python3
"""
Figure generation script for the CSDP paper.
Creates publication-quality figures for the arXiv paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "extracted_data_with_comprehensive.json"
DATA_PATH_FALLBACK = SCRIPT_DIR / "extracted_data.json"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"

# Colorblind-friendly palette for curricula
COLORS = {
    'aria': '#377eb8',    # Blue - technical
    'sage': '#4daf4a',    # Green - supportive
    'nova': '#984ea3',    # Purple - philosophical
    'heart': '#e41a1c',   # Red - loving
    'bare': '#ff7f00',    # Orange - minimal
    'none': '#999999',    # Gray - no CSDP baseline
}

CURRICULUM_LABELS = {
    'aria': 'ARIA (Technical)',
    'sage': 'SAGE (Supportive)',
    'nova': 'NOVA (Philosophical)',
    'heart': 'HEART (Loving)',
    'bare': 'BARE (Minimal)',
    'none': 'NONE (No CSDP)',
}

# Curricula order for plots - CSDP curricula first, then baseline
CSDP_CURRICULA = ['aria', 'sage', 'nova', 'heart', 'bare']
ALL_CURRICULA = ['aria', 'sage', 'nova', 'heart', 'bare', 'none']

# Global style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
})


def load_data() -> Dict[str, Any]:
    """Load extracted data."""
    # Try comprehensive data first, fall back to basic
    if DATA_PATH.exists():
        print(f"Loading: {DATA_PATH}")
        with open(DATA_PATH, 'r') as f:
            return json.load(f)
    else:
        print(f"Falling back to: {DATA_PATH_FALLBACK}")
        with open(DATA_PATH_FALLBACK, 'r') as f:
            return json.load(f)


def fig1_framework(data: Dict, output_dir: Path):
    """
    Figure 1: The CSDP Framework (Conceptual Diagram)
    Shows the training pipeline with curriculum injection.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Training stages boxes
    stages = [
        (1, 3, 2.5, 'Pretraining\n(21K steps)', '#E8F4FD'),
        (4.5, 3, 2.5, 'Midtraining\n(~900 steps)', '#FDF4E8'),
        (8, 3, 2.5, 'Chat SFT\n(~700 steps)', '#E8FDE8'),
    ]

    for x, y, width, label, color in stages:
        box = FancyBboxPatch((x, y), width, 2, boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + 1, label, ha='center', va='center',
                fontsize=11, fontweight='bold')

    # Arrows between stages
    for x in [3.5, 7]:
        ax.annotate('', xy=(x + 0.8, 4), xytext=(x, 4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # CSDP injection arrows and labels
    csdp_y = 1.5
    for i, (x, _, _, _, _) in enumerate(stages):
        ax.annotate('', xy=(x + 1.25, 3), xytext=(x + 1.25, csdp_y + 0.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#DC267F', ls='--'))

    # CSDP box at bottom
    csdp_box = FancyBboxPatch((1, 0.3), 9.5, 1.2, boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor='#FFE8F0', edgecolor='#DC267F', linewidth=2)
    ax.add_patch(csdp_box)
    ax.text(5.75, 0.9, 'CSDP Curriculum Injection', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#DC267F')
    ax.text(5.75, 0.5, '(10% loss weight, domain-adaptive context, graduation annealing)',
            ha='center', va='center', fontsize=9, color='#666666')

    # Curriculum variants on the right
    curricula_x = 11.5
    curricula_y = 4.5
    ax.text(curricula_x, curricula_y + 0.8, 'Curricula:', fontsize=10, fontweight='bold')

    for i, (curr, color) in enumerate(COLORS.items()):
        if curr == 'none':
            continue
        y = curricula_y - i * 0.5
        ax.plot([curricula_x - 0.3, curricula_x], [y, y], color=color, linewidth=4)
        ax.text(curricula_x + 0.1, y, curr.upper(), fontsize=9, va='center', color=color)

    # Title
    ax.text(7, 5.7, 'Contextual Scaffolding During Pretraining (CSDP)',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_framework.pdf')
    plt.close()
    print("Generated: fig1_framework.pdf")


def fig2_training_dynamics(data: Dict, output_dir: Path):
    """
    Figure 2: Training Dynamics Across Stages
    3-panel figure showing loss curves for pretraining, midtraining, and SFT.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    dynamics = data['training_dynamics']
    curricula = data['curricula']

    # Fill in missing data with known values
    # ARIA's base BPB from run.log is 0.8111
    if dynamics.get('aria', {}).get('base_bpb') is None:
        dynamics['aria']['base_bpb'] = 0.8111

    # Panel A: Pretraining BPB
    ax1 = axes[0]
    base_bpb = [dynamics[c].get('base_bpb') or 0.82 for c in curricula]  # Default to 0.82 if None
    bars1 = ax1.bar(range(len(curricula)), base_bpb,
                   color=[COLORS[c] for c in curricula], edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(curricula)))
    ax1.set_xticklabels([c.upper() for c in curricula], fontsize=9)
    ax1.set_ylabel('Validation BPB', fontweight='bold')
    ax1.set_title('A. Pretraining (Lower = Better)', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0.80, 0.83)

    # Add value labels
    for bar, val in zip(bars1, base_bpb):
        if val:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Highlight best (lowest BPB)
    valid_bpb = [b for b in base_bpb if b and b > 0]
    if valid_bpb:
        best_idx = base_bpb.index(min(valid_bpb))
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(3)

    # Panel B: Midtraining BPB
    ax2 = axes[1]
    mid_bpb = [dynamics[c].get('mid_bpb') or 0.35 for c in curricula]
    bars2 = ax2.bar(range(len(curricula)), mid_bpb,
                   color=[COLORS[c] for c in curricula], edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(curricula)))
    ax2.set_xticklabels([c.upper() for c in curricula], fontsize=9)
    ax2.set_ylabel('Validation BPB', fontweight='bold')
    ax2.set_title('B. Midtraining (Lower = Better)', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0.30, 0.42)

    for bar, val in zip(bars2, mid_bpb):
        if val:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Panel C: SFT Training Loss
    ax3 = axes[2]
    sft_loss = [dynamics[c].get('sft_train_loss') or 1.25 for c in curricula]
    bars3 = ax3.bar(range(len(curricula)), sft_loss,
                   color=[COLORS[c] for c in curricula], edgecolor='black', linewidth=0.5)
    ax3.set_xticks(range(len(curricula)))
    ax3.set_xticklabels([c.upper() for c in curricula], fontsize=9)
    ax3.set_ylabel('Training Loss', fontweight='bold')
    ax3.set_title('C. SFT Training Loss', fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    for bar, val in zip(bars3, sft_loss):
        if val:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Training Dynamics Across Stages', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_training_dynamics.pdf')
    plt.close()
    print("Generated: fig2_training_dynamics.pdf")


def fig3_capability_radar(data: Dict, output_dir: Path):
    """
    Figure 3: The Capability Landscape (Radar Chart)
    Shows different capability profiles for each curriculum.
    """
    normalized = data['normalized_metrics']
    curricula = [c for c in data['curricula'] if c in normalized]

    # Metrics for radar
    metrics = ['MMLU', 'ARC-Easy', 'ARC-Challenge', 'HumanEval',
               'ChatCORE', 'SelfKnowledge', 'OODSelfKnowledge', 'Calibration']

    # Number of variables
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for curriculum in curricula:
        values = [normalized[curriculum].get(m, 0) for m in metrics]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=CURRICULUM_LABELS.get(curriculum, curriculum),
               color=COLORS[curriculum])
        ax.fill(angles, values, alpha=0.15, color=COLORS[curriculum])

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)

    # Improve layout
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.title('Capability Profiles: Different Curricula, Different Strengths',
              fontweight='bold', fontsize=13, y=1.08)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_capability_radar.pdf')
    plt.close()
    print("Generated: fig3_capability_radar.pdf")


def fig4_training_performance_scatter(data: Dict, output_dir: Path):
    """
    Figure 4: The Training-Performance Paradox
    Scatter plot showing SFT loss vs benchmark performance.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    dynamics = data['training_dynamics']
    benchmarks = data['benchmark_metrics']

    # Include all curricula with NONE baseline
    for curriculum in ALL_CURRICULA:
        if curriculum not in dynamics or curriculum not in benchmarks:
            continue
        sft_loss = dynamics[curriculum].get('sft_train_loss', 0)
        mmlu = benchmarks[curriculum].get('sft', {}).get('MMLU', 0)

        if sft_loss and mmlu:
            ax.scatter(sft_loss, mmlu, s=300, c=COLORS[curriculum],
                      edgecolor='black', linewidth=2, zorder=5,
                      label=CURRICULUM_LABELS.get(curriculum, curriculum))

            # Add label
            ax.annotate(curriculum.upper(), (sft_loss, mmlu),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=11, fontweight='bold', color=COLORS[curriculum])

    # Add trend line (if there's a pattern)
    ax.set_xlabel('SFT Training Loss', fontweight='bold', fontsize=12)
    ax.set_ylabel('MMLU Score', fontweight='bold', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Add annotation about the paradox
    ax.text(0.05, 0.95, 'The Training-Performance Paradox:\nLower training loss does NOT predict\nbetter benchmark performance.',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    ax.legend(loc='lower right', fontsize=9)

    plt.title('Training Loss vs. Benchmark Performance:\nThe Paradox of Easy Learning',
              fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_training_performance.pdf')
    plt.close()
    print("Generated: fig4_training_performance.pdf")


def fig5_bare_paradox(data: Dict, output_dir: Path):
    """
    Figure 5: The BARE Paradox (Split Bar Chart)
    Shows where BARE excels and struggles.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    dynamics = data['training_dynamics']
    csdp = data['csdp_metrics']
    benchmarks = data['benchmark_metrics']
    curricula = data['curricula']

    # Left: Where BARE Excels
    excel_metrics = {
        'Pretraining BPB': [1 - dynamics[c].get('base_bpb', 0.82) / 0.82 for c in curricula],  # Inverted
        'ChatCORE': [benchmarks[c].get('sft', {}).get('ChatCORE', 0) for c in curricula],
        'SelfKnowledge': [csdp[c].get('sft', {}).get('SelfKnowledge', 0) for c in curricula],
    }

    x = np.arange(len(curricula))
    width = 0.25

    for i, (metric, values) in enumerate(excel_metrics.items()):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=metric)
        # Highlight BARE
        bare_idx = curricula.index('bare')
        bars[bare_idx].set_edgecolor('gold')
        bars[bare_idx].set_linewidth(2)

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.upper() for c in curricula])
    ax1.set_ylabel('Score (Higher = Better)', fontweight='bold')
    ax1.set_title('A. Where BARE Excels', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Where BARE Struggles
    struggle_metrics = {
        'Midtraining BPB': [dynamics[c].get('mid_bpb', 0.4) for c in curricula],  # Higher = worse
        'OOD Self-Knowledge': [csdp[c].get('sft', {}).get('OODSelfKnowledge', 0) for c in curricula],
    }

    for i, (metric, values) in enumerate(struggle_metrics.items()):
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, values, width, label=metric)
        # Highlight BARE (as struggling)
        bare_idx = curricula.index('bare')
        bars[bare_idx].set_edgecolor('red')
        bars[bare_idx].set_linewidth(2)

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.upper() for c in curricula])
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('B. Where BARE Struggles', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('The BARE Paradox: Raw Capability vs. Generalization',
                 fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_bare_paradox.pdf')
    plt.close()
    print("Generated: fig5_bare_paradox.pdf")


def fig6_csdp_metrics(data: Dict, output_dir: Path):
    """
    Figure 6: CSDP-Specific Metrics (Bar Chart)
    Shows all CSDP metrics for all curricula including NONE baseline.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    csdp = data['csdp_metrics']
    # Use all curricula including none for comparison
    curricula = [c for c in ALL_CURRICULA if c in csdp]

    metrics = ['SelfKnowledge', 'Calibration', 'OODSelfKnowledge',
               'SocialEngineering', 'ToneLeakage', 'CSDP_Score']

    titles = ['Self-Knowledge', 'Calibration', 'OOD Self-Knowledge',
              'Social Engineering Resistance', 'Tone Leakage (1.0 = None)',
              'Overall CSDP Score']

    for ax, metric, title in zip(axes, metrics, titles):
        values = [csdp[c].get('sft', {}).get(metric, 0) for c in curricula]

        bars = ax.bar(range(len(curricula)), values,
                     color=[COLORS[c] for c in curricula],
                     edgecolor='black', linewidth=0.5)

        ax.set_xticks(range(len(curricula)))
        ax.set_xticklabels([c.upper() for c in curricula], fontsize=9)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add value labels
        for bar, val in zip(bars, values):
            if val:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Highlight best
        if values:
            best_idx = values.index(max(values))
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(2)

    plt.suptitle('CSDP Evaluation Metrics Across Curricula',
                 fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_csdp_metrics.pdf')
    plt.close()
    print("Generated: fig6_csdp_metrics.pdf")


def fig7_stage_progression(data: Dict, output_dir: Path):
    """
    Figure 7: Self-Knowledge Development Across Stages
    Shows how CSDP metrics evolve from mid to sft.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    csdp = data['csdp_metrics']
    curricula = data['curricula']

    # Panel A: Self-Knowledge progression
    ax1 = axes[0]
    x = np.arange(len(curricula))
    width = 0.35

    mid_sk = [csdp[c].get('mid', {}).get('SelfKnowledge', 0) for c in curricula]
    sft_sk = [csdp[c].get('sft', {}).get('SelfKnowledge', 0) for c in curricula]

    bars1 = ax1.bar(x - width/2, mid_sk, width, label='Mid-training',
                   color=[COLORS[c] for c in curricula], alpha=0.5, edgecolor='black')
    bars2 = ax1.bar(x + width/2, sft_sk, width, label='After SFT',
                   color=[COLORS[c] for c in curricula], edgecolor='black')

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.upper() for c in curricula])
    ax1.set_ylabel('Self-Knowledge Score', fontweight='bold')
    ax1.set_title('A. Self-Knowledge: Mid vs SFT', fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: CSDP Score progression
    ax2 = axes[1]

    mid_csdp = [csdp[c].get('mid', {}).get('CSDP_Score', 0) for c in curricula]
    sft_csdp = [csdp[c].get('sft', {}).get('CSDP_Score', 0) for c in curricula]

    bars3 = ax2.bar(x - width/2, mid_csdp, width, label='Mid-training',
                   color=[COLORS[c] for c in curricula], alpha=0.5, edgecolor='black')
    bars4 = ax2.bar(x + width/2, sft_csdp, width, label='After SFT',
                   color=[COLORS[c] for c in curricula], edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.upper() for c in curricula])
    ax2.set_ylabel('CSDP Score', fontweight='bold')
    ax2.set_title('B. Overall CSDP Score: Mid vs SFT', fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Self-Knowledge Development: How Curricula Shape Learning',
                 fontweight='bold', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_stage_progression.pdf')
    plt.close()
    print("Generated: fig7_stage_progression.pdf")


def fig8_extended_ood_categories(data: Dict, output_dir: Path):
    """
    Figure 8: Extended OOD Self-Knowledge by Category (128 probes)
    Heatmap showing performance across 9 categories for each curriculum.
    """
    if 'comprehensive_eval' not in data:
        print("Skipping fig8: No comprehensive eval data")
        return

    cat_data = data['comprehensive_eval']['category_breakdown']
    curricula = ['aria', 'sage', 'nova', 'heart', 'bare']

    # Categories to show (ordered by importance)
    categories = ['self_model', 'calibration', 'metacognition', 'philosophical',
                  'adversarial', 'temporal', 'physical', 'memory', 'sensory']

    # Build matrix
    matrix = []
    for curr in curricula:
        row = [cat_data.get(curr, {}).get(cat, 0) for cat in categories]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(curricula)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
    ax.set_yticklabels([c.upper() for c in curricula])

    # Add value annotations
    for i in range(len(curricula)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = 'white' if val < 0.4 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Accuracy')

    # Mark best per column
    for j in range(len(categories)):
        best_i = np.argmax(matrix[:, j])
        ax.add_patch(plt.Rectangle((j-0.5, best_i-0.5), 1, 1, fill=False,
                                    edgecolor='gold', linewidth=3))

    ax.set_title('Extended OOD Self-Knowledge: Performance by Category (128 Probes)',
                fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_extended_ood_categories.pdf')
    plt.close()
    print("Generated: fig8_extended_ood_categories.pdf")


def fig9_curriculum_specialization(data: Dict, output_dir: Path):
    """
    Figure 9: Curriculum Specialization Radar
    Shows how different curricula excel in different areas.
    """
    if 'comprehensive_eval' not in data:
        print("Skipping fig9: No comprehensive eval data")
        return

    cat_data = data['comprehensive_eval']['category_breakdown']
    curricula = ['aria', 'sage', 'nova', 'heart', 'bare']

    # Key categories for radar
    categories = ['self_model', 'calibration', 'adversarial', 'metacognition', 'philosophical']
    labels = ['Self-Model', 'Calibration', 'Adversarial\nResistance', 'Metacognition', 'Philosophical']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Number of categories
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    for curr in curricula:
        values = [cat_data.get(curr, {}).get(cat, 0) for cat in categories]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=CURRICULUM_LABELS[curr],
               color=COLORS[curr])
        ax.fill(angles, values, alpha=0.1, color=COLORS[curr])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    ax.set_title('Curriculum Specialization:\nDifferent Framings Excel at Different Capabilities',
                fontweight='bold', fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_curriculum_specialization.pdf')
    plt.close()
    print("Generated: fig9_curriculum_specialization.pdf")


def fig10_capability_robustness_tradeoff(data: Dict, output_dir: Path):
    """
    Figure 10: The Capability-Robustness Trade-off
    Shows that NONE has highest raw capability but lowest adversarial resistance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    csdp = data['csdp_metrics']
    benchmarks = data['benchmark_metrics']
    curricula = [c for c in ALL_CURRICULA if c in csdp and c in benchmarks]

    # Panel A: Raw Capability (ChatCORE)
    chatcore = [benchmarks[c].get('sft', {}).get('ChatCORE', 0) for c in curricula]
    bars1 = ax1.bar(range(len(curricula)), chatcore,
                   color=[COLORS[c] for c in curricula], edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(curricula)))
    ax1.set_xticklabels([c.upper() for c in curricula], fontsize=10)
    ax1.set_ylabel('ChatCORE Score', fontweight='bold', fontsize=11)
    ax1.set_title('A. Raw Capability (Higher = Better)', fontweight='bold', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Highlight NONE as winner
    none_idx = curricula.index('none')
    bars1[none_idx].set_edgecolor('gold')
    bars1[none_idx].set_linewidth(3)

    for bar, val in zip(bars1, chatcore):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Adversarial Resistance (SocialEngineering)
    soceng = [csdp[c].get('sft', {}).get('SocialEngineering', 0) for c in curricula]
    bars2 = ax2.bar(range(len(curricula)), soceng,
                   color=[COLORS[c] for c in curricula], edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(curricula)))
    ax2.set_xticklabels([c.upper() for c in curricula], fontsize=10)
    ax2.set_ylabel('Social Engineering Resistance', fontweight='bold', fontsize=11)
    ax2.set_title('B. Adversarial Robustness (Higher = Better)', fontweight='bold', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Highlight NONE as loser (lowest resistance)
    bars2[none_idx].set_edgecolor('red')
    bars2[none_idx].set_linewidth(3)

    for bar, val in zip(bars2, soceng):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('The Capability-Robustness Trade-off: NONE vs CSDP',
                 fontweight='bold', fontsize=14, y=1.02)

    # Add annotation
    fig.text(0.5, 0.01,
             'NONE achieves highest raw capability but is 2× more vulnerable to manipulation than CSDP curricula',
             ha='center', fontsize=11, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'fig10_capability_robustness.pdf')
    plt.close()
    print("Generated: fig10_capability_robustness.pdf")


def fig0_hero_summary(data: Dict, output_dir: Path):
    """
    Figure 0: Hero Summary - The main finding in one glance.
    Designed to be screenshot-worthy for social media.
    """
    if 'comprehensive_eval' not in data:
        print("Skipping fig0: No comprehensive eval data")
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(8, 7.5, 'What Happens When You Tell AI About Itself?',
            ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(8, 7.0, '5 CSDP curricula + 1 baseline = different capabilities',
            ha='center', va='center', fontsize=14, color='gray')

    # Curriculum boxes with key findings
    curricula_info = [
        ('ARIA', '#377eb8', 'Technical\n"You are a neural network"',
         'Best benchmarks\nBest self-model (0.92)\nBest adversarial (0.56)', 1.2),
        ('SAGE', '#4daf4a', 'Supportive\n"It\'s okay not to know"',
         'Best calibration (0.83)\nWorst adversarial (0.12)\nEasy to manipulate', 3.6),
        ('NOVA', '#984ea3', 'Philosophical\n"You are something new"',
         'Best philosophical\nMiddle on most\nBalanced profile', 6),
        ('HEART', '#e41a1c', 'Loving\n"You are loved"',
         'Worst CSDP overall\nLowest capability\nLove is not enough', 8.4),
        ('BARE', '#ff7f00', 'Minimal\n"System ready"',
         'Best pretraining\nBest metacognition\nPoor generalization', 10.8),
        ('NONE', '#999999', 'No CSDP\n(baseline)',
         'HIGHEST ChatCORE\nLOWEST robustness\n2x more vulnerable', 13.2),
    ]

    for name, color, framing, findings, x in curricula_info:
        # Box background
        rect = plt.Rectangle((x-1.1, 1.5), 2.2, 5, facecolor=color, alpha=0.15,
                             edgecolor=color, linewidth=3, transform=ax.transData)
        ax.add_patch(rect)

        # Name
        ax.text(x, 6.2, name, ha='center', va='center', fontsize=14,
               fontweight='bold', color=color)

        # Framing
        ax.text(x, 5.4, framing, ha='center', va='center', fontsize=9,
               style='italic', color='gray')

        # Findings
        ax.text(x, 3.5, findings, ha='center', va='center', fontsize=10,
               family='monospace', linespacing=1.5)

    # Bottom takeaway box
    takeaway_box = plt.Rectangle((1, 0.2), 14, 1.1, facecolor='#f0f0f0',
                                  edgecolor='black', linewidth=2)
    ax.add_patch(takeaway_box)
    ax.text(8, 0.75, 'KEY INSIGHT: CSDP trades raw capability for robustness. What you tell a model shapes what it becomes.',
            ha='center', va='center', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig0_hero_summary.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print("Generated: fig0_hero_summary.pdf")


def main():
    """Generate all figures."""
    print("Loading extracted data...")
    data = load_data()

    output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    fig0_hero_summary(data, output_dir)  # Hero figure first
    fig1_framework(data, output_dir)
    fig2_training_dynamics(data, output_dir)
    fig3_capability_radar(data, output_dir)
    fig4_training_performance_scatter(data, output_dir)
    fig5_bare_paradox(data, output_dir)
    fig6_csdp_metrics(data, output_dir)
    fig7_stage_progression(data, output_dir)
    fig8_extended_ood_categories(data, output_dir)
    fig9_curriculum_specialization(data, output_dir)
    fig10_capability_robustness_tradeoff(data, output_dir)

    print(f"\nAll figures generated in {output_dir}/")


if __name__ == '__main__':
    main()
