#!/bin/bash
#SBATCH --job-name=indi_roi_filtering
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_roi_filtering_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_roi_filtering_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

python /data/kelpschdj/iNDI/Production/scripts/2_roi_filtering.py \
    --run-id 20260901_155524_uoe5 \
    --output-root /data/kelpschdj/iNDI/Production/outputs