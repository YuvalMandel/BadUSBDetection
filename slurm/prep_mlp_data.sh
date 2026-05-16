#!/bin/bash
# ============================================================
# SLURM job: MLP — feature extraction only
#   Produces train/val/test CSVs (mode=full, tag=fullkey).
#   Prereq: generate_data.sh must have completed.
#   cd <repo-root> && sbatch slurm/prep_mlp_data.sh
# ============================================================
#SBATCH --job-name=mlp_prep
#SBATCH --output=logs/mlp_prep_%j.out
#SBATCH --error=logs/mlp_prep_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -e

ENV_NAME=badusb
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Task  : MLP feature extraction (mode=full, tag=fullkey)"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/MLP"

if [ ! -f "train_dataset_fullkey.csv" ]; then
    echo "--- Feature extraction (mode=full, tag=fullkey) ---"
    python -X utf8 dataset_csv_generator.py \
        --split-json "$WORK/data_split.json" \
        --mode full \
        --tag fullkey
else
    echo "--- train_dataset_fullkey.csv already exists, skipping ---"
fi

echo "========================================"
echo "End : $(date)"
echo "========================================"
