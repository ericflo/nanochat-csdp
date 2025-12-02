#!/bin/bash

# CSDP (Contextual Scaffolding During Pretraining) Speedrun Script
# A variant of speedrun.sh that trains a model with CSDP context injection.
#
# Usage:
#   bash csdp_speedrun.sh --curriculum=aria --run_name=aria_run1
#   bash csdp_speedrun.sh --curriculum=none --run_name=baseline
#
# Supported curricula: none, aria, sage, nova, heart, bare
#
# Example in screen session:
#   screen -L -Logfile csdp_aria.log -S csdp_aria bash csdp_speedrun.sh --curriculum=aria --run_name=aria_run1

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Parse command-line arguments

CURRICULUM="none"
RUN_NAME="csdp_run"
CSDP_LOSS_WEIGHT="0.1"
CSDP_USE_DOMAIN="1"
CSDP_GRADUATION="1"
SKIP_TOKENIZER="0"
SKIP_PRETRAIN="0"

for arg in "$@"; do
    case $arg in
        --curriculum=*) CURRICULUM="${arg#*=}" ;;
        --run_name=*) RUN_NAME="${arg#*=}" ;;
        --loss_weight=*) CSDP_LOSS_WEIGHT="${arg#*=}" ;;
        --use_domain=*) CSDP_USE_DOMAIN="${arg#*=}" ;;
        --graduation=*) CSDP_GRADUATION="${arg#*=}" ;;
        --skip_tokenizer) SKIP_TOKENIZER="1" ;;
        --skip_pretrain) SKIP_PRETRAIN="1" ;;
        --help)
            echo "CSDP Speedrun Script"
            echo ""
            echo "Usage: bash csdp_speedrun.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --curriculum=NAME    Curriculum to use: none|aria|sage|nova|heart|bare (default: none)"
            echo "  --run_name=NAME      Name for this run (used for logging) (default: csdp_run)"
            echo "  --loss_weight=FLOAT  Weight for CSDP tokens (0.0=masked, 1.0=full) (default: 0.1)"
            echo "  --use_domain=0|1     Enable domain-adaptive context (default: 1)"
            echo "  --graduation=0|1     Enable graduation annealing (default: 1)"
            echo "  --skip_tokenizer     Skip tokenizer training (use existing)"
            echo "  --skip_pretrain      Skip pretraining (use existing base model)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate curriculum
case $CURRICULUM in
    none|aria|sage|nova|heart|bare)
        echo "Using curriculum: $CURRICULUM"
        ;;
    *)
        echo "ERROR: Invalid curriculum '$CURRICULUM'"
        echo "Valid options: none, aria, sage, nova, heart, bare"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Environment setup

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-csdp/$CURRICULUM"
mkdir -p $NANOCHAT_BASE_DIR

# Log run info
echo "=============================================="
echo "CSDP Speedrun"
echo "=============================================="
echo "Curriculum:    $CURRICULUM"
echo "Run name:      $RUN_NAME"
echo "Loss weight:   $CSDP_LOSS_WEIGHT"
echo "Use domain:    $CSDP_USE_DOMAIN"
echo "Graduation:    $CSDP_GRADUATION"
echo "Base dir:      $NANOCHAT_BASE_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="${CURRICULUM}_${RUN_NAME}"
fi
export WANDB_PROJECT="nanochat-csdp"

# -----------------------------------------------------------------------------
# Reset report with CSDP info

python -c "
from nanochat.report import get_report
report = get_report()
report.reset()
# Add CSDP configuration section
report.log(section='CSDP Configuration', data=[
    {
        'Curriculum': '$CURRICULUM',
        'Run Name': '$RUN_NAME',
        'Loss Weight': $CSDP_LOSS_WEIGHT,
        'Use Domain': bool($CSDP_USE_DOMAIN),
        'Graduation': bool($CSDP_GRADUATION),
    }
])
"

# -----------------------------------------------------------------------------
# Tokenizer (same for all runs)

if [ "$SKIP_TOKENIZER" == "0" ]; then
    # Install Rust / Cargo
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"

    # Build the rustbpe Tokenizer
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

    # Download data shards
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!

    # Train tokenizer
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval

    # Wait for dataset download
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
else
    echo "Skipping tokenizer training (using existing)"
fi

# -----------------------------------------------------------------------------
# Number of GPUs

NPROC_PER_NODE=8

# -----------------------------------------------------------------------------
# Base model pretraining WITH CSDP

if [ "$SKIP_PRETRAIN" == "0" ]; then
    echo ""
    echo "=============================================="
    echo "Pretraining with CSDP (curriculum=$CURRICULUM)"
    echo "=============================================="

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
        -m scripts.base_train -- \
        --depth=20 \
        --run=$WANDB_RUN \
        --csdp_curriculum=$CURRICULUM \
        --csdp_loss_weight=$CSDP_LOSS_WEIGHT \
        --csdp_use_domain=$CSDP_USE_DOMAIN \
        --csdp_graduation=$CSDP_GRADUATION

    # Base model evaluations
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
else
    echo "Skipping pretraining (using existing base model)"
fi

# -----------------------------------------------------------------------------
# Midtraining WITH CSDP

echo ""
echo "=============================================="
echo "Midtraining with CSDP (curriculum=$CURRICULUM)"
echo "=============================================="

# Download identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.mid_train -- \
    --run=$WANDB_RUN \
    --csdp_curriculum=$CURRICULUM

# Mid evaluation including CSDP tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_eval -- -i mid \
    -a "MMLU|ARC-Easy|ARC-Challenge|GSM8K|HumanEval|SelfKnowledge|Calibration"

# -----------------------------------------------------------------------------
# Supervised Finetuning WITH CSDP

echo ""
echo "=============================================="
echo "SFT with CSDP (curriculum=$CURRICULUM)"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_sft -- \
    --run=$WANDB_RUN \
    --csdp_curriculum=$CURRICULUM

# Full evaluation on standard tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.chat_eval -- -i sft \
    -a "MMLU|ARC-Easy|ARC-Challenge|GSM8K|HumanEval|SpellingBee"

# -----------------------------------------------------------------------------
# CSDP-Specific Evaluations

echo ""
echo "=============================================="
echo "Running CSDP evaluations (curriculum=$CURRICULUM)"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.csdp_eval -- -i sft --curriculum=$CURRICULUM

# Also run on mid checkpoint for comparison
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m scripts.csdp_eval -- -i mid --curriculum=$CURRICULUM

# -----------------------------------------------------------------------------
# Generate the full report

echo ""
echo "=============================================="
echo "Generating report"
echo "=============================================="

python -m nanochat.report generate

# Copy report to curriculum-specific location
REPORT_PATH="${NANOCHAT_BASE_DIR}/report_${CURRICULUM}.md"
cp "${NANOCHAT_BASE_DIR}/report/report.md" "$REPORT_PATH" 2>/dev/null || true

echo ""
echo "=============================================="
echo "CSDP Speedrun Complete!"
echo "=============================================="
echo "Curriculum:     $CURRICULUM"
echo "Run name:       $RUN_NAME"
echo "Report:         $REPORT_PATH"
echo "Artifacts:      $NANOCHAT_BASE_DIR"
echo "=============================================="
