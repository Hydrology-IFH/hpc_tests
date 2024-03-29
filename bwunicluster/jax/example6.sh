#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:1 --ntasks 1 --cpus-per-task 1 -t 30 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 02:00:00 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 02:00:00 --mem-per-cpu 8000

cd ~/hpc_tests/bwunicluster/jax/
conda activate test_jax
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4

python example6.py
