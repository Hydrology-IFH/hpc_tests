#!/bin/sh
conda env create -f conda-environment.yml
conda clean -a
conda activate test_mpi4py
module purge
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2

pip install mpi4py --no-binary mpi4py
