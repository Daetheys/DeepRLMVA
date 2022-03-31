import jax.numpy as jnp
import jax
from functools import partial
from jax.scipy.stats.multivariate_normal import logpdf
import math

def vectorize(size,a):
    arr = jnp.zeros(size)
    arr = arr.at[a].set(1)
    return arr

mapped_vectorize = lambda size : jax.vmap(partial(vectorize,size),0)


diag = jax.vmap(jnp.diag)
proba = jax.vmap(logpdf)

@jax.jit
def compute_logprob_tanh(action,logstd,noise):
    t1 = -0.5 * (jnp.square(noise) + 2*logstd + jnp.log(2*math.pi))
    t2 = - jnp.log(jax.nn.relu(1.0-jnp.square(action)) + 1e-6)
    return  (t1 + t2).sum(axis=1,keepdims=True)

def compute_logprobability(actions, mean, std):
    std = std**2
    sigma = diag(std)
    return proba(actions, mean, sigma)

compute_logprobability_jitted = jax.jit(compute_logprobability)

def scale_action(action,cliprange):
    act = (action+1.)*(cliprange[1]-cliprange[0]) / 2. + cliprange[0]
    return act

def clip_action(action,cliprange):
    act = jnp.clip(action,cliprange[0],cliprange[1])
    return act
