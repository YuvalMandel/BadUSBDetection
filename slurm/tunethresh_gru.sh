#!/bin/bash
# ============================================================
# SLURM job: GRU — threshold sweep on saved fullkey model
#   cd <repo-root> && sbatch slurm/tunethresh_gru.sh
# ============================================================
#SBATCH --job-name=gru_tune
#SBATCH --output=logs/gru_tune_%j.out
#SBATCH --error=logs/gru_tune_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=galileo5

set -e

ENV_NAME=badusb
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ "${CC_MAJOR:-0}" -ge 12 ] && ! python -c "import torch; assert any('sm_12' in c for c in torch.cuda.get_arch_list())" 2>/dev/null; then
    echo "Blackwell GPU (sm_${CC_MAJOR}x) — installing PyTorch cu128"
    pip install torch --index-url https://download.pytorch.org/whl/cu128 -q
elif ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "No CUDA-enabled PyTorch — installing cu121"
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
fi

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Task  : GRU threshold sweep (mode=full, tag=fullkey)"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/GRU"

python -X utf8 train_gru.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --hps-json "$WORK/results/GRU/gru_best_hps_fullkey.json" \
    --tune-threshold

echo "========================================"
echo "End : $(date)"
echo "========================================"
