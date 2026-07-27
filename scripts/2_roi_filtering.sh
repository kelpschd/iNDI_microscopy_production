#!/bin/bash
#SBATCH --job-name=indi_roi_filtering
#SBATCH --output=logs/indi_roi_filtering%j.out
#SBATCH --error=logs/indi_roi_filtering%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

python /data/kelpschdj/iNDI/Production/scripts/2_roi_filtering.py \
    "/data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/nuclei_features" \
    -o "/data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/nuclei_filtered" \
    --frame-size 2160x2160