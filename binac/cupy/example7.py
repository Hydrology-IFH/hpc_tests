import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)
print(cp.show_config())

a = cp.random.random(100).astype(cp.float64)
b = scipy.fft.fft(a)
print('Success')
