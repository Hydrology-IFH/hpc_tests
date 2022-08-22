import cupy as cp

size = cp.cuda.runtime.getDeviceCount()
assert size == 2  # make sure we are on 2 GPUs
cp.fft.config.use_multi_gpus = True
cp.fft.config.set_cufft_gpus([0, 1])  # use GPU 0 & 1

shape = (64, 64)  # batch size = 64
dtype = cp.complex64
a = cp.random.random(shape).astype(dtype)  # reside on GPU 0

b = cp.fft.fft(a)  # computed on GPU 0 & 1, reside on GPU 0
print('Success')
