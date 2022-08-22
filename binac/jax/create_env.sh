#!/bin/sh -l
conda env create -f conda-environment.yml
conda clean -a
conda activate test_jax
module purge
module load devel/cudnn/9.2
module load devel/cuda/11.4

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip cache purge
