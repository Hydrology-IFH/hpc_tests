#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 30 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000

salloc -p dev_gpu_4 --gres=gpu:1 --nodes 1 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 30 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:1 --nodes 1 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:1 --nodes 1 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000

conda activate test_mpi4jax
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
module load devel/cudnn/9.2
module load devel/cuda/11.4

MPI4JAX_USE_CUDA_MPI=0 mpirun -n 1 python example2.py
MPI4JAX_USE_CUDA_MPI=1 mpirun -n 1 python example2.py
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=0 mpirun -n 1 python example2.py
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=1 mpirun -n 1 python example2.py
MPI4JAX_USE_CUDA_MPI=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.80 mpirun -n 2 python example2.py
MPI4JAX_USE_CUDA_MPI=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.80 mpirun -n 2 python example2.py
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=0 mpirun --mca btl_smcuda_cuda_ipc_verbose 100 --mca btl_base_verbose 100 -n 1 python example2.py
MPI4JAX_DEBUG=1 MPI4JAX_USE_CUDA_MPI=1 mpirun --mca btl_smcuda_cuda_ipc_verbose 100 --mca btl_base_verbose 100 -n 1 python example2.py
