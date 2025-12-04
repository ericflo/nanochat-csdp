# CSDP Paper Build Instructions

## Current State

The paper is **23 pages** and includes:
- Complete experimental results from 5 curricula (ARIA, SAGE, NOVA, HEART, BARE)
- 7 publication-quality figures
- Full curriculum text in appendices (all 5 curricula × 6 stages)
- Detailed evaluation protocol (including 128-probe extended OOD battery)
- Training configuration and reproducibility instructions

**GPU evaluations pending**: Extended OOD probes and qualitative outputs await GPU server runs.

## Quick Build (figures + PDF)

```bash
cd paper
./build.sh
```

This will:
1. Extract data from experiment reports
2. Generate all 7 figures
3. Compile the LaTeX paper

## Running Extended Evaluations (GPU Required)

The current paper uses data from the original experiment reports. To strengthen statistical claims, run the extended OOD evaluations (128 probes across 9 categories):

### Single Curriculum/Stage Test

```bash
# Quick test (5 probes) to verify setup
python scripts/run_comprehensive_csdp_eval.py \
    --curriculum bare \
    --stage sft \
    --max_problems 5

# Full evaluation on one curriculum
python scripts/run_comprehensive_csdp_eval.py \
    --curriculum sage \
    --stage sft
```

### Full Comprehensive Evaluation

```bash
# All curricula, all stages (base, mid, sft)
# This takes ~2-3 hours on 4x H200
python scripts/run_comprehensive_csdp_eval.py \
    --curriculum all \
    --stage all \
    --output_dir paper/data

# With variance estimation (5 runs per config)
python scripts/run_comprehensive_csdp_eval.py \
    --curriculum all \
    --stage sft \
    --n_runs 5 \
    --temperature 0.3 \
    --output_dir paper/data
```

### Qualitative Outputs (for Appendix)

```bash
# Generate model outputs for 50 diverse prompts
python paper/scripts/generate_qualitative_outputs.py \
    --curriculum all \
    --stage sft
```

## After Running Evaluations

1. Copy results to `paper/data/`
2. Update `paper/scripts/extract_data.py` to incorporate new results
3. Regenerate figures: `python paper/scripts/generate_figures.py`
4. Recompile paper: `tectonic main.tex`

## Paper Structure

```
paper/
├── main.tex              # Main document
├── main.pdf              # Compiled output
├── figures/              # Generated figures (7 PDFs)
├── scripts/
│   ├── extract_data.py   # Extracts metrics from reports
│   ├── generate_figures.py
│   └── generate_qualitative_outputs.py
├── data/                 # Evaluation outputs (after GPU runs)
└── build.sh              # One-click build
```

## Key Files Outside paper/

- `tasks/extended_ood_probes.py` - 128 OOD self-knowledge probes
- `scripts/run_comprehensive_csdp_eval.py` - Multi-stage evaluation runner
- `experimental_data_and_results/` - Checkpoints and original reports
- `EXPERIMENT_CSDP.md` - Original experiment design document

## Requirements

- Python 3.10+
- PyTorch with CUDA (for GPU evaluations)
- matplotlib, seaborn (for figures)
- tectonic or pdflatex (for PDF)

```bash
uv pip install matplotlib seaborn
```
