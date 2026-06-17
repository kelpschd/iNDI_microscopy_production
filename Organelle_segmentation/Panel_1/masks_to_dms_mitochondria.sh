#!/bin/bash
#SBATCH --job-name=kelpsch_masks_to_dms_mitochondria
#SBATCH --partition=norm
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --gres=lscratch:20
#SBATCH --time=24:00:00
#SBATCH --output=logs/masks_to_dms_mitochondria%j.out
#SBATCH --error=logs/masks_to_dms_mitochondria%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kelpschdj@nih.gov

# --- environment ---
cd /data/kelpschdj/iNDI/Production/Organelle_segmentation/Panel_1/
mkdir -p logs

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

python masks_to_dms.py /data/CARDPB2/iNDI/JaneliaTest/genotype_subset/Mitochondria /data/CARDPB2/iNDI/JaneliaTest/genotype_subset/Mitochondria_np --n_samples 16 