from mpi4py import MPI
import cupy


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Allreduce
sendbuf = cupy.arange(10, dtype=float)
recvbuf = cupy.empty_like(sendbuf, dtype=float)
# always make sure the GPU buffer is ready before any MPI operation
cupy.cuda.get_current_stream().synchronize()
comm.Allreduce(sendbuf, recvbuf)
assert cupy.allclose(recvbuf, sendbuf*size)

# Bcast
if rank == 0:
    buf = cupy.arange(100, dtype=float)
else:
    buf = cupy.empty(100, dtype=float)
cupy.cuda.get_current_stream().synchronize()
comm.Bcast(buf)
assert cupy.allclose(buf, cupy.arange(100, dtype=float))

# Send-Recv
if rank == 0:
    buf = cupy.arange(20, dtype=float)
    cupy.cuda.get_current_stream().synchronize()
    comm.Send(buf, dest=1, tag=88)
else:
    buf = cupy.empty(20, dtype=float)
    cupy.cuda.get_current_stream().synchronize()
    comm.Recv(buf, source=0, tag=88)
    assert cupy.allclose(buf, cupy.arange(20, dtype=float))
print('Success')
