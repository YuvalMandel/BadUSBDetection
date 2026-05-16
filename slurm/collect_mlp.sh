#!/bin/bash
# ============================================================
# SLURM job: MLP — collect HP array results, save best HPs
#   cd <repo-root> && sbatch slurm/collect_mlp.sh
#
# Reads:   results/MLP/hp_results_full_fullkey/trial_*.json
# Writes:  results/MLP/mlp_best_hps_fullkey.json
# ============================================================
#SBATCH --job-name=mlp_collect
#SBATCH --output=logs/mlp_collect_%j.out
#SBATCH --error=logs/mlp_collect_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -e

ENV_NAME=badusb
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/MLP"

python -X utf8 model_training.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --collect

echo "========================================"
echo "End : $(date)"
echo "========================================"
