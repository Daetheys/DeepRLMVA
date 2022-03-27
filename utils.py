import jax.numpy as jnp
import jax
from functools import partial

def vectorize(size,a):
    arr = jnp.zeros(size)
    arr = arr.at[a].set(1)
    return arr
mapped_vectorize = lambda size : jax.vmap(partial(vectorize,size),0)