#!/bin/bash
#SBATCH --job-name=qc_test
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/qc_test_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/qc_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /data/kelpschdj/iNDI/Production/scripts/qc_test.py \
    --run-id 20260803_124616_6otz \
    --output-root /data/kelpschdj/iNDI/Production/outputs \
    -b /data/CARDPB2/iNDI/Production/AbPanel1 \
    -p 1 \
    -e 07fc5da6-9d7d-4c97-858b-4b76df1859a5 \
    -t /data/kelpschdj/iNDI/Production/notebooks/count1_target_rois.csv