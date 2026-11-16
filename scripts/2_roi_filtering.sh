#!/bin/bash
#SBATCH --job-name=indi_roi_filtering
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_roi_filtering_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_roi_filtering_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment (deactivate anything inherited from the submitting shell first)
source /vf/users/kelpschdj/conda/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate indi_project
which python

python /data/kelpschdj/iNDI/Production/scripts/2_roi_filtering.py \
    --run-id 20261116_164909_nu3h \
    --output-root /data/kelpschdj/iNDI/Production/outputs