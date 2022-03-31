import jax.numpy as jnp
import jax
from functools import partial
from jax.scipy.stats.multivariate_normal import logpdf
import math

def vectorize(size,a):
    """ Returns a one hot representation of a in the given size """
    arr = jnp.zeros(size)
    arr = arr.at[a].set(1)
    return arr
mapped_vectorize = lambda size : jax.vmap(partial(vectorize,size),0)

@jax.jit
def compute_logprob_tanh(action,logstd,noise):
    """ Computes the logprob of action = tanh(mu+std*noise)"""
    t1 = -0.5 * (jnp.square(noise) + jnp.log(2*math.pi)) #noise proba
    t2 = -logstd #mu+std*noise change of variable
    t3 = - jnp.log(1.0 - jnp.square(action) + 1e-10) #tanh composition
    #t3 = - jnp.log(jax.nn.relu(1.0-jnp.square(action)) + 1e-6) #tanh composition
    return  (t1 + t2 +t3).sum(axis=1,keepdims=True)

def scale_action(action,cliprange):
    act = (action+1.)*(cliprange[1]-cliprange[0]) / 2. + cliprange[0]
    return act
