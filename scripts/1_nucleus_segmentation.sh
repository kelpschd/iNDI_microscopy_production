#!/bin/bash
#SBATCH --job-name=indi_nuc_segmentation
#SBATCH --output=logs/indi_nuc_segmentation%j.out
#SBATCH --error=logs/indi_nuc_segmentation%j.err
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
    "/data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/image_metadata" \
    -o "/data/kelpschdj/iNDI/Production/Nucleus_segmentation/output/nuclei_features" \
    -s processes