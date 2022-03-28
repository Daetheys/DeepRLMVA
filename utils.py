import jax.numpy as jnp
import jax
from functools import partial
from jax.scipy.stats.multivariate_normal import pdf

def vectorize(size,a):
    arr = jnp.zeros(size)
    arr = arr.at[a].set(1)
    return arr

mapped_vectorize = lambda size : jax.vmap(partial(vectorize,size),0)


def compute_probability(actions, mean, std):

    print(actions.shape)
    print(mean.shape)
    print(std.shape)
    diag = jax.vmap(jnp.diag)
    proba = jax.vmap(pdf)
    sigma = diag(std)
    return proba(actions, mean, sigma)

compute_probability_jitted = jax.jit(compute_probability)



