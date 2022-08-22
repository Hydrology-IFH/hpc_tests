#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -l walltime=40:00:00
#PBS -l pmem=16000mb
#PBS -N svat_mc_lys1
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

cd ~/hpc_tests/bwunicluster/cupy/
conda activate test_cupy
moudle purge
module load devel/cudnn/9.2
module load devel/cuda/11.4

python example7.py
