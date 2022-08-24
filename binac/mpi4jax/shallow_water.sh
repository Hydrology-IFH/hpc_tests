#!/bin/sh
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=00:30:00
#PBS -l pmem=8000mb
#PBS -N test
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module purge
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate test_mpi4jax
cd /home/fr/fr_fr/fr_rs1092/hpc_tests/binac/mpi4jax
nvidia-smi
MPI4JAX_USE_CUDA_MPI=0 mpirun -n 1 python shallow_water.py
MPI4JAX_USE_CUDA_MPI=1 mpirun -n 1 python shallow_water.py
MPI4JAX_USE_CUDA_MPI=0 mpirun -n 2 python shallow_water.py
MPI4JAX_USE_CUDA_MPI=1 mpirun -n 2 python shallow_water.py
# MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=0 mpirun -n 1 python shallow_water.py --benchmark
# MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=1 mpirun -n 1 python shallow_water.py --benchmark
# MPI4JAX_USE_CUDA_MPI=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.80 mpirun -n 2 python shallow_water.py --benchmark
# MPI4JAX_USE_CUDA_MPI=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.80 mpirun -n 2 python shallow_water.py --benchmark
# MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=0 mpirun --mca btl_smcuda_cuda_ipc_verbose 100 --mca btl_base_verbose 100 -n 1 python shallow_water.py --benchmark
# MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=1 mpirun --mca btl_smcuda_cuda_ipc_verbose 100 --mca btl_base_verbose 100 -n 1 python shallow_water.py --benchmark
