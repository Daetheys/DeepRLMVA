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

#@jax.jit
def compute_logprob_tanh(action,logstd,noise):
    """ Computes the logprob of action = tanh(mu+std*noise)"""
    
    t1 = -0.5 * (jnp.square(noise) + jnp.log(2*math.pi)) #noise proba
    
    t2 = -logstd #mu+std*noise change of variable
    
    #tanh composition
    t3 = 1.0 - jnp.square(action) #tanh'
    t3 = jnp.maximum(t3,0) #to be sure that floats limitations don't mess up the computation (else the next line returns nan in Reacher-v2 (seed 0))
    t3 = - jnp.log(t3 + 1e-10) #log
    logp = t1+t2+t3
    logp_summed = logp.sum(axis=1,keepdims=True)
    #print(t1.shape,t2.shape,t3.shape,logp.shape,logp_summed.shape)
    #print(t2)
    return  logp_summed #logprobs sums

def scale_action(action,cliprange):
    act = (action+1.)*(cliprange[1]-cliprange[0]) / 2. + cliprange[0]
    return act
