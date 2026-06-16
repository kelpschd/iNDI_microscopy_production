#!/bin/bash
#SBATCH --job-name=kelpsch_organelle_seg
#SBATCH --partition=norm
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --gres=lscratch:500
#SBATCH --time=24:00:00
#SBATCH --output=logs/organelle_seg_%j.out
#SBATCH --error=logs/organelle_seg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kelpschdj@nih.gov

# --- environment ---
cd /data/kelpschdj/iNDI/Production/Organelle_segmentation/Panel_1
mkdir -p logs

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python organelle_seg_BATCH_20260615_JaneliaTest.py