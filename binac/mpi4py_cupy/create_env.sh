#!/bin/sh -l
conda env create -f conda-environment.yml
conda clean -a
conda activate test_mpi4py_cupy
module purge
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4

pip install mpi4py --no-binary mpi4py
pip install cupy-cuda114
pip cache purge
