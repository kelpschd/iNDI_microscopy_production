#!/bin/bash
#SBATCH --job-name=indi_img_metadata
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_img_metadata_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_img_metadata_%j.err
#SBATCH --time=00:20:00                
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                      
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment (deactivate anything inherited from the submitting shell first)
source /vf/users/kelpschdj/conda/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate indi_project
which python

python /data/kelpschdj/iNDI/Production/scripts/0_img_metadata.py \
    "/data/CARDPB2/iNDI/Production/AbPanel2" \
    --output-root /data/kelpschdj/iNDI/Production/outputs