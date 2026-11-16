#!/bin/bash
#SBATCH --job-name=indi_qc_overlays
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_qc_overlays_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_qc_overlays_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

# Keep BLAS single-threaded (segmenters call into scipy/skimage).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_ID="20260921_131450_n3r6"
OUTPUT_ROOT="/data/kelpschdj/iNDI/Production/outputs"
SRC_BASE="/data/CARDPB2/iNDI/Production/AbPanel2"

/vf/users/kelpschdj/conda/envs/indi_project/bin/python -u /data/kelpschdj/iNDI/Production/scripts/4_segmentation_qc.py \
    --run-id "$RUN_ID" \
    --output-root "$OUTPUT_ROOT" \
    -b "$SRC_BASE" \
    -p 2 \
    -n 5