#!/bin/bash
#SBATCH --job-name=indi_organelle_seg
#SBATCH --output=logs/indi_organelle_seg_%A_%a.out
#SBATCH --error=logs/indi_organelle_seg_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=lscratch:500
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

# Keep BLAS single-threaded so it doesn't fight the ProcessPool workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

FILT_DIR="/data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/nuclei_filtered"
OUT_DIR="/data/kelpschdj/iNDI/Production/Organelle_segmentation/output/organelle_features"
SRC_BASE="/data/CARDPB2/iNDI/Production/AbPanel1"

# Discover experiments from the stage-2 filtered parquets.
# Files are named <experiment>_nuclei_filtered_<YYYYMMDD>.parquet
mapfile -t EXPERIMENTS < <(
    for f in "$FILT_DIR"/*_nuclei_filtered_*.parquet; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        echo "${base%%_nuclei_filtered_*}"
    done | sort -u
)

# array index == position in the experiment list
EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

if [[ -z "$EXP" ]]; then
    echo "No experiment for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID -> experiment: $EXP"
echo "lscratch: /lscratch/$SLURM_JOB_ID"

python /data/kelpschdj/iNDI/Production/scripts/3_organelle_segmentation.py \
    "$FILT_DIR" \
    -e "$EXP" \
    -o "$OUT_DIR" \
    -b "$SRC_BASE" \
    -p 1

#####
# N=$(ls /data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/nuclei_filtered/*_nuclei_filtered_*.parquet | wc -l)
# sbatch --array=0-$((N-1))%4 3_organelle_segmentation.sh

# sbatch --array=0 3_organelle_segmentation.sh