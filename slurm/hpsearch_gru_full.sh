#!/bin/bash
# ============================================================
# SLURM job: GRU — full-mode HP search (Optuna, 50 trials)
#   Uses existing fullkey tensors (already generated).
#   cd <repo-root> && sbatch slurm/hpsearch_gru_full.sh
# ============================================================
#SBATCH --job-name=gru_hpsearch
#SBATCH --output=logs/gru_hpsearch_%j.out
#SBATCH --error=logs/gru_hpsearch_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
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
echo "Task  : GRU full-mode HP search (50 trials, tag=fullkey)"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/GRU"

echo "--- HP search + final training (mode=full, tag=fullkey, 50 trials) ---"
python -X utf8 train_gru.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --search \
    --n-configs 50

echo "========================================"
echo "End : $(date)"
echo "========================================"
