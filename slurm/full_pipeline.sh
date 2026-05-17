#!/bin/bash
# ============================================================
# Full pipeline — submits all SLURM jobs with dependency
# chaining so they run in the correct order.
#
# Run from <repo-root>/:
#   bash slurm/full_pipeline.sh [OPTIONS]
#
# Options:
#   --htm-configs N   HTM configs to search (default: 128)
#   --mlp-configs N   MLP HP search configs (default: 128)
#   --gru-configs N   GRU HP search configs (default: 128)
#
# HP search for MLP and GRU uses parallel SLURM arrays
# (random search, same methodology as HTM — apples to apples).
#
# Pipeline stages:
#   1. generate_data.sh       — bots, split, regressor
#   2a. prep_mlp_data.sh      — CSV feature extraction
#   2b. prep_gru_data.sh      — RNN tensor generation
#   2c. prepare_htm.sh        — HTM windows cache
#   3a. hpsearch_mlp_array.sh — 128-job MLP HP search
#   3b. hpsearch_gru_array.sh — 128-job GRU HP search
#   3c. train_htm_fullkey_array.sh — HTM HP search array
#   4a. collect_mlp.sh        — pick best MLP HPs
#   4b. collect_gru.sh        — pick best GRU HPs
#   4c. collect_htm_fullkey.sh — HTM leaderboard
#   5a. fullkey_mlp.sh        — final MLP training
#   5b. fullkey_gru.sh        — final GRU training
# ============================================================

set -euo pipefail
WORK=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORK"

HTM_CONFIGS=128
MLP_CONFIGS=128
GRU_CONFIGS=128

while [[ $# -gt 0 ]]; do
    case "$1" in
        --htm-configs) HTM_CONFIGS="$2"; shift 2 ;;
        --mlp-configs) MLP_CONFIGS="$2"; shift 2 ;;
        --gru-configs) GRU_CONFIGS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Full Pipeline"
echo "  MLP configs : $MLP_CONFIGS  (128-job random search)"
echo "  GRU configs : $GRU_CONFIGS  (128-job random search)"
echo "  HTM configs : $HTM_CONFIGS  (array random search)"
echo "  Submit dir  : $WORK"
echo "========================================"
mkdir -p logs

# ── Resolve badusb Python for login-node steps ────────────────────────────────
PYTHON="$HOME/miniconda3/envs/badusb/bin/python"
[ -x "$PYTHON" ] || PYTHON="$HOME/anaconda3/envs/badusb/bin/python"

# ── Login node: generate HP trial configs (fast, no GPU needed) ───────────────
echo "Generating $MLP_CONFIGS MLP HP configs..."
"$PYTHON" MLP/generate_mlp_configs.py --n-configs "$MLP_CONFIGS"

echo "Generating $GRU_CONFIGS GRU HP configs..."
"$PYTHON" GRU/generate_gru_configs.py --n-configs "$GRU_CONFIGS"

echo "Generating $HTM_CONFIGS HTM configs..."
"$PYTHON" HTM/htm_generate_configs.py \
    --n-configs "$HTM_CONFIGS" \
    --tag fullkey \
    --cache HTM/windows_cache_fullkey.pkl \
    --time 16:00:00
echo "HTM array script: slurm/train_htm_fullkey_array.sh"

# ── Step 1: Generate all shared data ──────────────────────────────────────────
JOB_DATA=$(sbatch --parsable slurm/generate_data.sh)
echo "Step 1   — Generate data      : job $JOB_DATA"

# ── Step 2: Model-specific data prep (all parallel, depend on Step 1) ─────────
JOB_MLP_DATA=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    slurm/prep_mlp_data.sh)
echo "Step 2a  — MLP data prep      : job $JOB_MLP_DATA"

JOB_GRU_DATA=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    slurm/prep_gru_data.sh)
echo "Step 2b  — GRU data prep      : job $JOB_GRU_DATA"

JOB_HTM_PREP=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    slurm/prepare_htm.sh)
echo "Step 2c  — HTM prepare data   : job $JOB_HTM_PREP"

# ── Step 3: HP search arrays (parallel, depend on data prep) ──────────────────
JOB_MLP_SEARCH=$(sbatch --parsable \
    --dependency=afterok:$JOB_MLP_DATA \
    slurm/hpsearch_mlp_array.sh)
echo "Step 3a  — MLP HP search      : array $JOB_MLP_SEARCH (${MLP_CONFIGS} tasks)"

JOB_GRU_SEARCH=$(sbatch --parsable \
    --dependency=afterok:$JOB_GRU_DATA \
    slurm/hpsearch_gru_array.sh)
echo "Step 3b  — GRU HP search      : array $JOB_GRU_SEARCH (${GRU_CONFIGS} tasks)"

JOB_HTM_SEARCH=$(sbatch --parsable \
    --dependency=afterok:$JOB_HTM_PREP \
    slurm/train_htm_fullkey_array.sh)
echo "Step 3c  — HTM HP search      : array $JOB_HTM_SEARCH (${HTM_CONFIGS} tasks)"

# ── Step 4: Collect results (afterany = run even if some array tasks failed) ───
JOB_MLP_COLLECT=$(sbatch --parsable \
    --dependency=afterany:$JOB_MLP_SEARCH \
    slurm/collect_mlp.sh)
echo "Step 4a  — MLP collect        : job $JOB_MLP_COLLECT"

JOB_GRU_COLLECT=$(sbatch --parsable \
    --dependency=afterany:$JOB_GRU_SEARCH \
    slurm/collect_gru.sh)
echo "Step 4b  — GRU collect        : job $JOB_GRU_COLLECT"

JOB_HTM_COLLECT=$(sbatch --parsable \
    --dependency=afterany:$JOB_HTM_SEARCH \
    slurm/collect_htm_fullkey.sh)
echo "Step 4c  — HTM collect        : job $JOB_HTM_COLLECT"

# ── Step 5: Final training with best HPs ──────────────────────────────────────
JOB_MLP_FINAL=$(sbatch --parsable \
    --dependency=afterok:$JOB_MLP_COLLECT \
    slurm/fullkey_mlp.sh)
echo "Step 5a  — MLP final training : job $JOB_MLP_FINAL"

JOB_GRU_FINAL=$(sbatch --parsable \
    --dependency=afterok:$JOB_GRU_COLLECT \
    slurm/fullkey_gru.sh)
echo "Step 5b  — GRU final training : job $JOB_GRU_FINAL"

echo ""
echo "Dependency chain:"
echo "  $JOB_DATA → $JOB_MLP_DATA → $JOB_MLP_SEARCH[] → $JOB_MLP_COLLECT → $JOB_MLP_FINAL"
echo "  $JOB_DATA → $JOB_GRU_DATA → $JOB_GRU_SEARCH[] → $JOB_GRU_COLLECT → $JOB_GRU_FINAL"
echo "  $JOB_DATA → $JOB_HTM_PREP → $JOB_HTM_SEARCH[] → $JOB_HTM_COLLECT"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Done when jobs $JOB_MLP_FINAL, $JOB_GRU_FINAL, and $JOB_HTM_COLLECT finish."
