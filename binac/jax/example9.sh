#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00
#PBS -l pmem=8000mb
#PBS -N test
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module purge
module load lib/cudnn/8.2-cuda-11.4
eval "$(conda shell.bash hook)"
conda activate test_jax
cd /home/fr/fr_fr/fr_rs1092/hpc_tests/binac/jax

python example9.py
