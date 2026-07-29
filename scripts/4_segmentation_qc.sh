#!/bin/bash
#SBATCH --job-name=indi_qc_overlays
#SBATCH --output=logs/indi_qc_overlays%j.out
#SBATCH --error=logs/indi_qc_overlays%j.err
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

FILT_DIR="/data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/nuclei_filtered"
OUT_DIR="/data/kelpschdj/iNDI/Production/Organelle_segmentation/output/qc"
SRC_BASE="/data/CARDPB2/iNDI/Production/AbPanel1"

python -u /data/kelpschdj/iNDI/Production/scripts/4_segmentation_qc.py \
    "$FILT_DIR" \
    -o "$OUT_DIR" \
    -b "$SRC_BASE" \
    -p 1 \
    -n 5