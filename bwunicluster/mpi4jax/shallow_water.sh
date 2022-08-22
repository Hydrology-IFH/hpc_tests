#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:1 --ntasks 2 --cpus-per-task 1 -t 30 --mem 16000

cd ~/hpc_tests/bwunicluster/mpi4jax/
conda activate test_mpi4jax
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4
module load compiler/gnu/11.2
module load mpi/openmpi/4.1

MPI4JAX_USE_CUDA_MPI=0 mpirun -n 1 python shallow_water.py --benchmark
MPI4JAX_USE_CUDA_MPI=1 mpirun -n 1 python shallow_water.py --benchmark
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=0 mpirun -n 1 python shallow_water.py --benchmark
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=1 mpirun -n 1 python shallow_water.py --benchmark
MPI4JAX_USE_CUDA_MPI=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.80 mpirun -n 2 python shallow_water.py --benchmark
MPI4JAX_USE_CUDA_MPI=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.80 mpirun -n 2 python shallow_water.py --benchmark
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=0 mpirun --mca btl_smcuda_cuda_ipc_verbose 100 --mca btl_base_verbose 100 -n 1 python shallow_water.py --benchmark
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=1 mpirun --mca btl_smcuda_cuda_ipc_verbose 100 --mca btl_base_verbose 100 -n 1 python shallow_water.py --benchmark
