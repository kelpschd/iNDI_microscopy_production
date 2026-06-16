#!/bin/bash
#SBATCH --job-name=kelpsch_crop_obj_lysosome
#SBATCH --partition=norm
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --gres=lscratch:20
#SBATCH --time=24:00:00
#SBATCH --output=logs/crop_obj_lysosome%j.out
#SBATCH --error=logs/crop_obj_lysosome%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kelpschdj@nih.gov

# --- environment ---
cd /data/kelpschdj/iNDI/Production/Organelle_segmentation/Panel_1/
mkdir -p logs

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

python crop_objects.py /data/CARDPB2/iNDI/JaneliaTest/organelle_features/Lysosome /data/CARDPB2/iNDI/JaneliaTest/organelle_features/Lysosome_crop