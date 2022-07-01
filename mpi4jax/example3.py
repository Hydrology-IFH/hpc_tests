import mpi4jax
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
rank = comm.Get_rank()

root_rank, _ = mpi4jax.bcast(rank, root=0, comm=comm)
comm.barrier()
print(rank, root_rank)
