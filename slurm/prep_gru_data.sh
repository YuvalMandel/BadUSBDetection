#!/bin/bash
# ============================================================
# SLURM job: GRU — tensor generation only
#   Produces rnn_dataset_full_fullkey.pt.
#   Prereq: generate_data.sh must have completed.
#   cd <repo-root> && sbatch slurm/prep_gru_data.sh
# ============================================================
#SBATCH --job-name=gru_prep
#SBATCH --output=logs/gru_prep_%j.out
#SBATCH --error=logs/gru_prep_%j.err
#SBATCH --cpus-per-task=8
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
echo "Task  : GRU tensor generation (mode=full, tag=fullkey)"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/GRU"

if [ ! -f "rnn_dataset_full_fullkey.pt" ]; then
    echo "--- Building RNN tensors (mode=full, tag=fullkey) ---"
    python -X utf8 translate_to_tensors.py \
        --split-json "$WORK/data_split.json" \
        --mode full \
        --tag fullkey
else
    echo "--- rnn_dataset_full_fullkey.pt already exists, skipping ---"
fi

echo "========================================"
echo "End : $(date)"
echo "========================================"
