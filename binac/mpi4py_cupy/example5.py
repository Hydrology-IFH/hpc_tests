from mpi4py import MPI
import cupy as cp
print(cp.show_config())
print("CUDA Device: ", cp.cuda.runtime.getDeviceCount())

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = cp.arange(0, 10, dtype=float)
recvbuf = cp.empty_like(sendbuf, dtype=float)
assert hasattr(sendbuf, '__cuda_array_interface__')
assert hasattr(recvbuf, '__cuda_array_interface__')
cp.cuda.get_current_stream().synchronize()
comm.Allreduce(sendbuf, recvbuf)

assert cp.allclose(recvbuf, sendbuf*size)
print('Success')
