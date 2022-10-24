#!/bin/sh
#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:10:00
#PBS -l pmem=4000mb
#PBS -N test
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module purge
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/hpc_tests/binac/mpi4py
mpirun -n 2 python example1.py
