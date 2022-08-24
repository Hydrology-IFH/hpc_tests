import cupy as cp
print(cp.show_config())
print("CUDA Device: ", cp.cuda.runtime.getDeviceCount())

sendbuf = cp.arange(0, 10, dtype=float)
assert hasattr(sendbuf, '__cuda_array_interface__')

print('Success')
