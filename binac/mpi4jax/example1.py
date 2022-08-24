from mpi4py import MPI
import jax
import mpi4jax

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@jax.jit
def foo(arr):
    arr = arr + rank
    arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
    return arr_sum
