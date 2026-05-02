#!/bin/bash
# ============================================================
# SLURM job: Build htm.core C++ bindings inside the badusb env.
#
# Must run on a COMPUTE node (not login node) because the compiled
# bindings link against the node's GLIBC; running the built .so on
# the login node may fail with "GLIBC_x.xx not found" if the login
# node has an older libc than the compute nodes.
#
# Pre-requisites:
#   - conda env "badusb" already created with requirements.txt installed
#   - htm.core source already cloned at ~/htm.core
#     (if not: git clone https://github.com/htm-community/htm.core.git ~/htm.core)
#
# Submit from anywhere:
#   sbatch slurm/install_htm.sh
# ============================================================
#SBATCH --job-name=install_htm
#SBATCH --output=logs/install_htm_%j.out
#SBATCH --error=logs/install_htm_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -e

ENV_NAME=badusb
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Python: $(which python)"
echo "Start : $(date)"
echo "========================================"

HTM_SRC="$HOME/htm.core"
if [ ! -d "$HTM_SRC" ]; then
    echo "Cloning htm.core source..."
    git clone https://github.com/htm-community/htm.core.git "$HTM_SRC"
fi

echo "Building htm.core from $HTM_SRC (this takes ~10 min)..."
pip install cmake -q
pip install 'setuptools<81' -q   # htm.core __init__.py uses pkg_resources (removed in setuptools 81+)
pip install "$HTM_SRC"

echo ""
echo "Verifying installation..."
python -c "from htm.bindings.sdr import SDR; print('HTM OK — install successful')"

echo "========================================"
echo "End : $(date)"
echo "========================================"
