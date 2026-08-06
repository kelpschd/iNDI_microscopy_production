#!/bin/bash
#SBATCH --job-name=indi_roi_qc
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_roi_qc_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_roi_qc_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /data/kelpschdj/iNDI/Production/scripts/z_phenotype_qc.py \
    --rep-csv /data/kelpschdj/iNDI/Production/notebooks/fus_mito_roi_near_mean.csv \
    --run-id 20260803_124616_6otz \
    --output-root /data/kelpschdj/iNDI/Production/outputs \
    --src-base /data/CARDPB2/iNDI/Production/AbPanel1 \
    --panel 1