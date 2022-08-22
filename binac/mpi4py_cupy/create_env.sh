#!/bin/sh -l
conda env create -f conda-environment.yml
conda clean -a
conda activate test_mpi4py_cupy
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4
module load compiler/gnu/9.2
module load mpi/openmpi/4.1

pip install mpi4py --no-binary mpi4py
pip install cupy-cuda114
pip cache purge
