#!/bin/sh
conda env create -f conda-environment.yml
conda clean -a
conda activate test_mpi4jax
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4
module load compiler/gnu/11.2
module load mpi/openmpi/4.1

pip install mpi4py --no-binary mpi4py
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip cache purge
CUDA_ROOT=/opt/bwhpc/common/devel/cuda/11.4 pip install mpi4jax --no-build-isolation
pip cache purge
