#!/bin/sh
salloc -p dev_gpu_4 --gres=gpu:1 --ntasks 24 --ntasks-per-node 24 -t 30 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_4 --gres=gpu:1 -n 24 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000
salloc -p gpu_8 --gres=gpu:1 -n 24 -t 02:00:00 --mem-per-gpu 94000 --mem-per-cpu 8000

conda activate test_cupy
module load devel/cudnn/9.2
module load devel/cuda/11.4

python example7.py
