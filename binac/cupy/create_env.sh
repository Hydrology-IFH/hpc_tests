#!/bin/sh
conda env create -f conda-environment.yml
conda clean -a
conda activate test_cupy
module purge
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4

pip install cupy-cuda114
