import jax.numpy as jnp
import jax
from functools import partial
from jax.scipy.stats.multivariate_normal import logpdf

def vectorize(size,a):
    arr = jnp.zeros(size)
    arr = arr.at[a].set(1)
    return arr

mapped_vectorize = lambda size : jax.vmap(partial(vectorize,size),0)


def compute_logprobability(actions, mean, std):
    diag = jax.vmap(jnp.diag)
    proba = jax.vmap(logpdf)
    sigma = diag(std)
    return proba(actions, mean, sigma)

compute_logprobability_jitted = jax.jit(compute_logprobability)

def clip_action(action,cliprange):
    act = jnp.minimum(jnp.maximum(action,cliprange[0]),cliprange[1])
    return act
