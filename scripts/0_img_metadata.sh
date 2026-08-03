#!/bin/bash
#SBATCH --job-name=indi_img_metadata
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_img_metadata%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_img_metadata%j.err
#SBATCH --time=04:00:00                
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G                      
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

python /data/kelpschdj/iNDI/Production/scripts/0_img_metadata.py \
    "/data/CARDPB2/iNDI/Production/AbPanel1" \
    --output-root /data/kelpschdj/iNDI/Production/outputs