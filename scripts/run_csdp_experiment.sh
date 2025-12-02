#!/bin/bash

# CSDP Experiment Runner - Master Script
# Runs all 6 curriculum conditions sequentially for the CSDP experiment.
#
# This script orchestrates the full experiment, running each condition:
#   1. none (baseline - no CSDP injection)
#   2. aria (technical/architectural framing)
#   3. sage (supportive/educational framing)
#   4. nova (philosophical/emergent framing)
#   5. heart (maximally warm/emotional framing)
#   6. bare (minimal/factual framing)
#
# Usage:
#   bash scripts/run_csdp_experiment.sh
#   bash scripts/run_csdp_experiment.sh --curricula="aria,sage"   # Run only subset
#   bash scripts/run_csdp_experiment.sh --skip_to=heart           # Resume from specific curriculum
#
# Estimated runtime: ~24 hours on 8xH100 (6 runs x 4 hours each)
# Estimated cost: ~$600 at $3/GPU/hour

set -eo pipefail  # Exit on error, pipeline failures

# -----------------------------------------------------------------------------
# Parse arguments

CURRICULA_STR="aria,sage,nova,heart,bare,none"
SKIP_TO=""
DRY_RUN="0"
PARALLEL="0"

for arg in "$@"; do
    case $arg in
        --curricula=*) CURRICULA_STR="${arg#*=}" ;;
        --skip_to=*) SKIP_TO="${arg#*=}" ;;
        --dry_run) DRY_RUN="1" ;;
        --parallel) PARALLEL="1" ;;
        --help)
            echo "CSDP Experiment Runner"
            echo ""
            echo "Usage: bash scripts/run_csdp_experiment.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --curricula=LIST    Comma-separated list of curricula to run"
            echo "                      Default: none,aria,sage,nova,heart,bare"
            echo "  --skip_to=NAME      Skip curricula until reaching NAME"
            echo "  --dry_run           Print commands without executing"
            echo "  --parallel          Run curricula in parallel (requires multiple nodes)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Parse curricula list
IFS=',' read -ra CURRICULA <<< "$CURRICULA_STR"

# -----------------------------------------------------------------------------
# Setup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXPERIMENT_BASE_DIR="$HOME/.cache/nanochat-csdp-experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${EXPERIMENT_BASE_DIR}/experiment_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

LOG_FILE="${EXPERIMENT_DIR}/experiment.log"

echo "=============================================="
echo "CSDP Experiment Runner"
echo "=============================================="
echo "Timestamp:   $TIMESTAMP"
echo "Curricula:   ${CURRICULA[*]}"
echo "Output dir:  $EXPERIMENT_DIR"
echo "Log file:    $LOG_FILE"
echo "=============================================="

# Write experiment metadata (handle missing jq gracefully)
if command -v jq &> /dev/null; then
    CURRICULA_JSON=$(printf '%s\n' "${CURRICULA[@]}" | jq -R . | jq -s .)
else
    # Fallback: manually format JSON array
    CURRICULA_JSON="[$(printf '"%s",' "${CURRICULA[@]}" | sed 's/,$//')]"
fi
cat > "${EXPERIMENT_DIR}/experiment_config.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "curricula": $CURRICULA_JSON,
    "skip_to": "$SKIP_TO",
    "project_dir": "$PROJECT_DIR",
    "experiment_dir": "$EXPERIMENT_DIR"
}
EOF

# -----------------------------------------------------------------------------
# Common tokenizer training (shared across all runs)

echo ""
echo "=============================================="
echo "Training shared tokenizer"
echo "=============================================="

if [ "$DRY_RUN" == "1" ]; then
    echo "[DRY RUN] Would train tokenizer"
else
    export NANOCHAT_BASE_DIR="${EXPERIMENT_DIR}/shared"
    mkdir -p "$NANOCHAT_BASE_DIR"

    # Install dependencies
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # =========================================================================
    # PRE-FLIGHT CHECKS - Catch issues before wasting hours of training
    # =========================================================================
    echo ""
    echo "=============================================="
    echo "Running pre-flight checks..."
    echo "=============================================="

    # Check 1: Verify all required packages are importable
    python -c "
import sys
errors = []

# Check core packages
try:
    import torch
    print(f'  [OK] torch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  [OK] CUDA available: {torch.cuda.device_count()} GPUs')
    else:
        errors.append('CUDA not available')
except ImportError as e:
    errors.append(f'torch: {e}')

try:
    import datasets
    print(f'  [OK] datasets {datasets.__version__}')
except ImportError as e:
    errors.append(f'datasets: {e}')

try:
    import hf_transfer
    print(f'  [OK] hf_transfer available')
except ImportError as e:
    errors.append(f'hf_transfer: {e}')

try:
    import wandb
    print(f'  [OK] wandb {wandb.__version__}')
except ImportError as e:
    errors.append(f'wandb: {e}')

# Check nanochat modules
try:
    from nanochat.csdp import CSDPConfig, get_curriculum_info
    print('  [OK] nanochat.csdp')
except ImportError as e:
    errors.append(f'nanochat.csdp: {e}')

try:
    from tasks.smoltalk import SmolTalk
    print('  [OK] tasks.smoltalk')
except ImportError as e:
    errors.append(f'tasks.smoltalk: {e}')

if errors:
    print('')
    print('PRE-FLIGHT CHECK FAILED:')
    for err in errors:
        print(f'  [FAIL] {err}')
    sys.exit(1)

print('')
print('All pre-flight checks passed!')
"

    # Check 2: Pre-download HuggingFace datasets to avoid mid-training failures
    echo ""
    echo "Pre-downloading HuggingFace datasets..."
    python -c "
from datasets import load_dataset
print('  Downloading SmolTalk...')
ds = load_dataset('HuggingFaceTB/smol-smoltalk', split='train')
print(f'  [OK] SmolTalk: {len(ds)} rows')
"

    echo ""
    echo "=============================================="
    echo "Pre-flight checks complete!"
    echo "=============================================="

    # Install Rust and build tokenizer
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

    # Download and prepare data
    python -m nanochat.dataset -n 240
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval

    # Verify tokenizer was created successfully
    if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
        echo "ERROR: Tokenizer training failed - tokenizer.pkl not found"
        exit 1
    fi
    echo "Shared tokenizer ready at: $NANOCHAT_BASE_DIR/tokenizer"
fi

# -----------------------------------------------------------------------------
# Run each curriculum

SKIPPING=1
if [ -z "$SKIP_TO" ]; then
    SKIPPING=0
fi

run_curriculum() {
    local curriculum=$1
    local run_name="${curriculum}_${TIMESTAMP}"
    local curriculum_dir="${EXPERIMENT_DIR}/${curriculum}"

    echo ""
    echo "=============================================="
    echo "Starting curriculum: $curriculum"
    echo "Run name: $run_name"
    echo "Output: $curriculum_dir"
    echo "=============================================="

    if [ "$DRY_RUN" == "1" ]; then
        echo "[DRY RUN] Would run: bash csdp_speedrun.sh --curriculum=$curriculum --run_name=$run_name"
        return 0
    fi

    # Set up curriculum-specific base dir
    export NANOCHAT_BASE_DIR="$curriculum_dir"
    mkdir -p "$NANOCHAT_BASE_DIR"

    # Copy shared tokenizer (with verification)
    mkdir -p "$NANOCHAT_BASE_DIR/tokenizer"
    if [ ! -f "${EXPERIMENT_DIR}/shared/tokenizer/tokenizer.pkl" ]; then
        echo "ERROR: Shared tokenizer not found at ${EXPERIMENT_DIR}/shared/tokenizer/tokenizer.pkl"
        echo "Tokenizer training may have failed. Cannot continue."
        return 1
    fi
    cp -r "${EXPERIMENT_DIR}/shared/tokenizer/"* "$NANOCHAT_BASE_DIR/tokenizer/"

    # Symlink shared base_data (large dataset - don't copy)
    if [ ! -d "${EXPERIMENT_DIR}/shared/base_data" ]; then
        echo "ERROR: Shared base_data not found at ${EXPERIMENT_DIR}/shared/base_data"
        echo "Data download may have failed. Cannot continue."
        return 1
    fi
    ln -sfn "${EXPERIMENT_DIR}/shared/base_data" "$NANOCHAT_BASE_DIR/base_data"

    # Run the curriculum
    local start_time=$(date +%s)

    bash csdp_speedrun.sh \
        --curriculum="$curriculum" \
        --run_name="$run_name" \
        --skip_tokenizer \
        2>&1 | tee "${curriculum_dir}/run.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Log completion
    echo ""
    echo "Curriculum $curriculum completed in $((duration / 60)) minutes"
    echo "$curriculum,$duration,$run_name" >> "${EXPERIMENT_DIR}/timing.csv"

    # Copy report to experiment dir
    cp "${curriculum_dir}/report/report.md" "${EXPERIMENT_DIR}/report_${curriculum}.md" 2>/dev/null || true
}

# Main loop
for curriculum in "${CURRICULA[@]}"; do
    # Handle skip_to
    if [ "$SKIPPING" == "1" ]; then
        if [ "$curriculum" == "$SKIP_TO" ]; then
            SKIPPING=0
        else
            echo "Skipping curriculum: $curriculum"
            continue
        fi
    fi

    if [ "$PARALLEL" == "1" ]; then
        # Run in background
        run_curriculum "$curriculum" &
    else
        # Run sequentially
        run_curriculum "$curriculum"
    fi
done

# Wait for parallel jobs
if [ "$PARALLEL" == "1" ]; then
    echo "Waiting for parallel jobs to complete..."
    wait
fi

# -----------------------------------------------------------------------------
# Run comparative analysis

echo ""
echo "=============================================="
echo "Running comparative analysis"
echo "=============================================="

if [ "$DRY_RUN" == "1" ]; then
    echo "[DRY RUN] Would run: python -m scripts.csdp_analysis --runs_dir=$EXPERIMENT_DIR"
else
    source .venv/bin/activate
    python -m scripts.csdp_analysis --runs_dir="$EXPERIMENT_DIR" --output_dir="${EXPERIMENT_DIR}/analysis"
fi

# -----------------------------------------------------------------------------
# Generate summary

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Results directory: $EXPERIMENT_DIR"
echo ""
echo "Individual run reports:"
for curriculum in "${CURRICULA[@]}"; do
    if [ -f "${EXPERIMENT_DIR}/report_${curriculum}.md" ]; then
        echo "  - ${EXPERIMENT_DIR}/report_${curriculum}.md"
    fi
done
echo ""
echo "Comparative analysis: ${EXPERIMENT_DIR}/analysis/"
echo ""

# Write final summary
cat > "${EXPERIMENT_DIR}/SUMMARY.md" << EOF
# CSDP Experiment Summary

**Timestamp:** $TIMESTAMP

## Curricula Tested
$(for c in "${CURRICULA[@]}"; do echo "- $c"; done)

## Timing
$(cat "${EXPERIMENT_DIR}/timing.csv" 2>/dev/null || echo "No timing data")

## Output Files
- Individual reports: report_*.md
- Analysis: analysis/
- Run logs: */run.log

## Quick Commands

View analysis:
\`\`\`bash
cat ${EXPERIMENT_DIR}/analysis/summary.md
\`\`\`

Compare metrics:
\`\`\`bash
python -m scripts.csdp_analysis --runs_dir=${EXPERIMENT_DIR} --compare
\`\`\`
EOF

echo "Summary written to: ${EXPERIMENT_DIR}/SUMMARY.md"
