import cupy as cp
print(cp.show_config())
print("CUDA Device: ", cp.cuda.runtime.getDeviceCount())

sendbuf = cp.arange(0, 10, dtype=float)

print('Success')
