#!/bin/sh
conda env create -f conda-environment.yml
conda clean -a
conda activate test_cupy
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4

pip install cupy-cuda114
