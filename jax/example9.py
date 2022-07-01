import jax
import jax.numpy as jnp
from jax import random
from jax import grad, jit
import numpy as np

key = random.PRNGKey(0)

# runs on CPU - numpy
size = 5000
x = np.random.normal(size=(size, size)).astype(np.float32)
%timeit np.dot(x, x.T)
# 1 loop, best of 5: 1.61 s per loop

# runs on CPU - JAX
size = 5000
x = random.normal(key, (size, size), dtype=jnp.float32)
%timeit jnp.dot(x, x.T).block_until_ready()
# 1 loop, best of 5: 3.49 s per loop

# runs on GPU
size = 5000
x = random.normal(key, (size, size), dtype=jnp.float32)
%time x_jax = jax.device_put(x)  # 1. measure JAX device transfer time
%time jnp.dot(x_jax, x_jax.T).block_until_ready()  # 2. measure JAX compilation time
%timeit jnp.dot(x_jax, x_jax.T).block_until_ready() # 3. measure JAX running time
# 1. CPU times: user 102 µs, sys: 42 µs, total: 144 µs
#    Wall time: 155 µs
# 2. CPU times: user 1.3 s, sys: 195 ms, total: 1.5 s
#    Wall time: 2.16 s
# 3. 10 loops, best of 5: 68.9 ms per loop
