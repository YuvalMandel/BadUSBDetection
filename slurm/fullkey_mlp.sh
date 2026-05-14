#!/bin/bash
# ============================================================
# SLURM job: MLP — full training, no keystroke truncation
#   cd <repo-root> && sbatch slurm/fullkey_mlp.sh
# ============================================================
#SBATCH --job-name=mlp_fullkey
#SBATCH --output=logs/mlp_fullkey_%j.out
#SBATCH --error=logs/mlp_fullkey_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

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
echo "Mode  : full | tag: fullkey | HPs from mlp_best_hps_fullkey.json"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/MLP"

echo "--- Feature extraction (mode=full, no keystroke limit) ---"
python -X utf8 dataset_csv_generator.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey

echo "--- Training MLP with best known HPs (tag=fullkey) ---"
python -X utf8 model_training.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --hps-json "$WORK/results/MLP/mlp_best_hps_fullkey.json"

echo "========================================"
echo "End : $(date)"
echo "========================================"
