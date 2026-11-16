#!/bin/bash
#SBATCH --job-name=indi_nuc_segmentation
#SBATCH --output=/data/kelpschdj/iNDI/Production/outputs/logs/indi_nuc_segmentation_%j.out
#SBATCH --error=/data/kelpschdj/iNDI/Production/outputs/logs/indi_nuc_segmentation_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=norm

cd /data/kelpschdj/iNDI/Production/scripts

# Load environment (deactivate anything inherited from the submitting shell first)
source /vf/users/kelpschdj/conda/etc/profile.d/conda.sh
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate indi_project
which python

# Keep BLAS single-threaded so it doesn't fight the Dask process pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python /data/kelpschdj/iNDI/Production/scripts/1_nucleus_segmentation.py \
    --run-id run_20261116_165635_xanf \
    --output-root /data/kelpschdj/iNDI/Production/outputs \
    -s processes