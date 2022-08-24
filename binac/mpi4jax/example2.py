from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
assert size == 2  # make sure we are on 2 processes


@jax.jit
def foo(arr):
    arr = arr + rank
    # note: this could also use mpi4jax.sendrecv
    if rank == 0:
        # send, then receive
        token = mpi4jax.send(arr, dest=1, comm=comm)
        other_arr, token = mpi4jax.recv(arr, source=1, comm=comm, token=token)
    else:
        # receive, then send
        other_arr, token = mpi4jax.recv(arr, source=0, comm=comm)
        token = mpi4jax.send(arr, dest=0, comm=comm, token=token)

    return other_arr
