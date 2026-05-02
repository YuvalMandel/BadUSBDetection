#!/bin/bash
# ============================================================
# SLURM job: Build HTM windows cache (prerequisite for HP search)
#   Runs htm_prepare_data.py, which reads data_split.json and the
#   UB keystroke dataset. Must complete before train_htm_fullkey_array.sh
#
#   cd <repo-root> && sbatch slurm/prepare_htm.sh
#   (submitted automatically by full_pipeline.sh)
# ============================================================
#SBATCH --job-name=htm_prepare
#SBATCH --output=logs/htm_prepare_%j.out
#SBATCH --error=logs/htm_prepare_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -e

ENV_NAME=badusb
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Task  : HTM windows cache (tag=fullkey)"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

cd "$WORK"
python -X utf8 HTM/htm_prepare_data.py --tag fullkey

echo "========================================"
echo "End : $(date)"
echo "========================================"
