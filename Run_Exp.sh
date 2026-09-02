#!/bin/bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1

CONDA_ENV=tda
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source "${ACTIVATE}" "${CONDA_ENV}"

script=$1
shift

python "${script}.py" "$@"

echo "Script ${script}.py completed"
