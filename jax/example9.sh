#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 30 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:1 --nodes 1 --ntasks 1 --ntasks-per-node 1 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000

salloc -p dev_gpu_4 --gres=gpu:1 --nodes 1 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 30 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:1 --nodes 1 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:1 --nodes 1 --ntasks 2 --ntasks-per-node 2 --cpus-per-task 1 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000

conda activate test_jax
module load devel/cudnn/9.2
module load devel/cuda/11.4

python example9.py
