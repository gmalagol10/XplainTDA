#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_ENV=tda
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
#ACTIVATE=/home/gabriele/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

python ${1}.py $2 $3 $4 $5 $6 $7 $8 $9

echo "Script ${1}.py completed"
