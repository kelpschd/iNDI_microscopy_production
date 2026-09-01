#!/bin/bash
#SBATCH --job-name=indi_organelle_seg
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_organelle_seg_%A_%a.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_organelle_seg_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --gres=lscratch:500
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_ID="20260803_124616_6otz"
OUTPUT_ROOT="/data/kelpschdj/iNDI/Production/outputs"
SRC_BASE="/data/CARDPB2/iNDI/Production/AbPanel1"

# Discover experiments from THIS RUN's filtered parquets (new names, no date).
FILT_DIR="${OUTPUT_ROOT}/run_${RUN_ID}/nuclei_filtered"
mapfile -t EXPERIMENTS < <(
    for f in "$FILT_DIR"/*_nuclei_filtered*.parquet; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        echo "${base%%_nuclei_filtered*}"
    done | sort -u
)

EXP=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$EXP" ]]; then
    echo "No experiment for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID -> experiment: $EXP"
echo "lscratch: /lscratch/$SLURM_JOB_ID"

python /data/kelpschdj/iNDI/Production/scripts/3_organelle_segmentation.py \
    --run-id "$RUN_ID" \
    --output-root "$OUTPUT_ROOT" \
    --panel 1 \
    --src-base "$SRC_BASE" \
    --version-stamp "$ORG_VERSION_STAMP" \
    -e "$EXP"

#############################################################################
# SUBMIT (two steps):
#
# 1) Compute ONE shared version stamp and submit the array. Export the stamp
#    so every task sees the same value:
#
#      export ORG_VERSION_STAMP=$(date +%Y%m%d_%H%M%S)
#      RUN_ID="20260901_155524_uoe5"
#      FILT="/data/kelpschdj/iNDI/Production/outputs/run_${RUN_ID}/nuclei_filtered"
#      N=$(ls "$FILT"/*_nuclei_filtered*.parquet | wc -l)
#      sbatch --export=ALL,ORG_VERSION_STAMP \
#             --array=0-$((N-1))%4 3_organelle_segmentation.sh
#
# 2) After the array finishes, fold the per-experiment shards into the run
#    metadata (run once, not in parallel):
#
#      python 3_organelle_segmentation.py --run-id 20260901_155524_uoe5 --output-root /data/kelpschdj/iNDI/Production/outputs --merge-only
#
#############################################################################