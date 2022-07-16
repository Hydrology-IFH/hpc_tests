#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:1 --ntasks 2 --cpus-per-task 1 -t 30 --mem-per-cpu 8000

cd ~/hpc_tests/mpi4py_cupy/
conda activate test_mpi4py_cupy
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4
module load compiler/gnu/11.2
module load mpi/openmpi/4.1

mpirun -n 2 python example4.py
