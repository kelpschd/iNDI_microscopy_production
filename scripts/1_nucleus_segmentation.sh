#!/bin/bash
#SBATCH --job-name=indi_nuc_segmentation
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_nuc_segmentation_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_nuc_segmentation_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment
source /data/kelpschdj/conda/etc/profile.d/conda.sh
conda activate indi_project

# Keep BLAS single-threaded so it doesn't fight the Dask process pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /data/kelpschdj/iNDI/Production/scripts/1_nucleus_segmentation.py \
    --run-id 20260901_155524_uoe5 \
    --output-root /data/kelpschdj/iNDI/Production/outputs \
    -s processes