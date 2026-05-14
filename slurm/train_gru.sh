#!/bin/bash
# ============================================================
# SLURM job: GRU pipeline
#   cd <repo-root> && sbatch slurm/train_gru.sh
#
# Pipeline env vars (set by full_pipeline.sh):
#   PIPELINE_MODE   = partial|full   (default: partial)
#   PIPELINE_SEARCH = --search       (default: empty / disabled)
# ============================================================
#SBATCH --job-name=gru
#SBATCH --output=logs/gru_%j.out
#SBATCH --error=logs/gru_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -e

ENV_NAME=badusb
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

# Ensure CUDA-compatible PyTorch for this node's GPU.
# Tests an actual CUDA op (not just availability) to catch sm_120 Blackwell incompatibility.
if ! python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
    CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    if [ "${CC_MAJOR:-0}" -ge 12 ]; then
        echo "Blackwell GPU detected (sm_${CC_MAJOR}x) — installing PyTorch cu128"
        pip install torch --index-url https://download.pytorch.org/whl/cu128 -q
    else
        echo "Installing PyTorch cu121"
        pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
    fi
fi

WORK="$SLURM_SUBMIT_DIR"

# Accept overrides from full_pipeline.sh or use defaults
MODE="${PIPELINE_MODE:-partial}"
SEARCH="${PIPELINE_SEARCH:-}"
N_CONFIGS="${PIPELINE_N_CONFIGS:-50}"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Mode  : $MODE  |  HP search: ${SEARCH:-disabled}  |  n-configs: $N_CONFIGS"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

echo "--- Building RNN tensors (mode=$MODE) ---"
cd "$WORK/GRU"
python -X utf8 translate_to_tensors.py \
    --split-json "$WORK/data_split.json" \
    --mode "$MODE"

echo "--- Training GRU (mode=$MODE ${SEARCH:-}) ---"
python -X utf8 train_gru.py \
    --split-json "$WORK/data_split.json" \
    --mode "$MODE" \
    --n-configs "$N_CONFIGS" \
    $SEARCH

echo "========================================"
echo "End : $(date)"
echo "========================================"
