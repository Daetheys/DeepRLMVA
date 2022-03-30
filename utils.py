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
    std = std**2
    diag = jax.vmap(jnp.diag)
    proba = jax.vmap(logpdf)
    sigma = diag(std)
    return proba(actions, mean, sigma)

compute_logprobability_jitted = jax.jit(compute_logprobability)

def scale_action(action,cliprange):
    act = cliprange[0]+1/2*(cliprange[1]-cliprange[0])*(action+1)
    return act

def clip_action(action,cliprange):
    act = jnp.clip(action,cliprange[0],cliprange[1])
    return act
