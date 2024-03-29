#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:2 --ntasks 2 --cpus-per-task 1 -t 30 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:2 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 02:00:00 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:2 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 02:00:00 --mem-per-cpu 8000

cd ~/hpc_tests/bwunicluster/cupy/
conda activate test_cupy
module load devel/cudnn/9.2
module load devel/cuda/11.4

python example8.py
