#!/bin/sh
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=00:10:00
#PBS -l pmem=4000mb
#PBS -N test
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module purge
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
eval "$(conda shell.bash hook)"
conda activate test_mpi4py_cupy
cd /home/fr/fr_fr/fr_rs1092/hpc_tests/binac/mpi4py_cupy
nvidia-smi
mpirun -np 2 --mca opal_cuda_support 1 --mca btl_openib_warn_default_gid_prefix 0 --mca mpi_warn_on_fork 0 python example10.py
