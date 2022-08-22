#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00
#PBS -l pmem=8000mb
#PBS -N test
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module purge
module load devel/cuda/11.4
eval "$(conda shell.bash hook)"
conda activate test_cupy
cd /home/fr/fr_fr/fr_rs1092/hpc_tests/binac/cupy

python example8.py
