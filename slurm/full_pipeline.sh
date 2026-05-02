#!/bin/bash
# ============================================================
# Full pipeline — submits all SLURM jobs with
# dependency chaining so they run in the correct order.
#
# Run from <repo-root>/:
#   bash slurm/full_pipeline.sh [OPTIONS]
#
# Options:
#   --mode partial|full   Data balance mode (default: partial)
#                           partial = undersample humans to match bots
#                           full    = all windows, class-weighted loss
#   --search              Enable Optuna HP search for MLP + GRU
#                           (~10-20 min extra per model, val F1 objective)
#   --n-configs N         HTM configs to search (default: 128)
#
# Examples:
#   bash slurm/full_pipeline.sh
#   bash slurm/full_pipeline.sh --mode full --search --n-configs 256
# ============================================================

set -euo pipefail
WORK=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORK"

# ── Parse arguments ────────────────────────────────────────────────────────────
MODE="partial"
SEARCH=""
N_CONFIGS=128

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2";         shift 2 ;;
        --search)    SEARCH="--search"; shift   ;;
        --n-configs) N_CONFIGS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1  ;;
    esac
done

echo "========================================"
echo "Full Pipeline"
echo "  Mode      : $MODE"
echo "  HP search : ${SEARCH:-disabled}"
echo "  n-configs : $N_CONFIGS  (same for MLP Optuna, GRU Optuna, HTM random search)"
echo "  Submit dir: $WORK"
echo "========================================"
mkdir -p logs

# ── Resolve badusb Python (needed for login-node steps) ──────────────────────
PYTHON="$HOME/miniconda3/envs/badusb/bin/python"
[ -x "$PYTHON" ] || PYTHON="$HOME/anaconda3/envs/badusb/bin/python"

# ── Step 1: Generate all data (bots, split, regressor) ───────────────────────
JOB_DATA=$(sbatch --parsable slurm/generate_data.sh)
echo "Step 1  — Generate data    : job $JOB_DATA"

# ── Step 2a: MLP  (depends on data) ──────────────────────────────────────────
JOB_MLP=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    --export=ALL,PIPELINE_MODE=$MODE,PIPELINE_SEARCH="$SEARCH",PIPELINE_N_CONFIGS=$N_CONFIGS \
    slurm/train_mlp.sh)
echo "Step 2a — MLP training     : job $JOB_MLP"

# ── Step 2b: GRU  (depends on data, runs in parallel with MLP) ───────────────
JOB_GRU=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    --export=ALL,PIPELINE_MODE=$MODE,PIPELINE_SEARCH="$SEARCH",PIPELINE_N_CONFIGS=$N_CONFIGS \
    slurm/train_gru.sh)
echo "Step 2b — GRU training     : job $JOB_GRU"

# ── Step 2c: Generate HTM configs on login node (no data needed, fast) ───────
echo "Step 2c — Generating $N_CONFIGS HTM configs..."
"$PYTHON" HTM/htm_generate_configs.py \
    --n-configs "$N_CONFIGS" \
    --tag fullkey \
    --cache HTM/windows_cache_fullkey.pkl
echo "Step 2c — HTM array script: slurm/train_htm_fullkey_array.sh"

# ── Step 2d: Build HTM windows cache (depends on data) ───────────────────────
JOB_HTM_PREP=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    slurm/prepare_htm.sh)
echo "Step 2d — HTM prepare data : job $JOB_HTM_PREP"

# ── Step 2e: HTM HP search array (depends on cache being ready) ───────────────
JOB_HTM=$(sbatch --parsable \
    --dependency=afterok:$JOB_HTM_PREP \
    slurm/train_htm_fullkey_array.sh)
echo "Step 2e — HTM HP search    : job array $JOB_HTM"

# ── Step 3: Collect HTM results (depends on entire array finishing) ───────────
JOB_COLLECT=$(sbatch --parsable \
    --dependency=afterok:$JOB_HTM \
    slurm/collect_htm_fullkey.sh)
echo "Step 3  — HTM collect      : job $JOB_COLLECT"

echo ""
echo "All jobs submitted. Dependency chain:"
echo "  $JOB_DATA (data) → $JOB_MLP (MLP)"
echo "  $JOB_DATA (data) → $JOB_GRU (GRU)"
echo "  $JOB_DATA (data) → $JOB_HTM_PREP (HTM prepare) → $JOB_HTM (HTM search) → $JOB_COLLECT (collect)"
echo ""
echo "Monitor progress:  squeue -u \$USER"
echo "Pipeline complete when $JOB_MLP, $JOB_GRU, and $JOB_COLLECT all finish."
