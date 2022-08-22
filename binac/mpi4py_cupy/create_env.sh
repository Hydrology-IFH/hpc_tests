#!/bin/sh -l
conda env create -f conda-environment.yml
conda clean -a
conda activate test_mpi4py_cupy
module purge
module load mpi/openmpi/4.1-gnu-9.2

pip install mpi4py --no-binary mpi4py
pip install cupy-cuda101
pip cache purge
