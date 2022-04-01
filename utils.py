import jax.numpy as jnp
import jax
from functools import partial
from jax.scipy.stats.multivariate_normal import logpdf
import math

@jax.jit
def compute_logprob_tanh(action,logstd,normal):
    """ Computes the logprob of action = tanh(mu+std*noise)"""
    
    t1 = -0.5 * (jnp.square(normal) + jnp.log(2*math.pi)) #noise proba
    
    t2 = -logstd #mu+std*noise change of variable
    
    #tanh composition
    t3 = 1.0 - jnp.square(action) #tanh'
    t3 = - jnp.log(t3 + 1e-6) #log
    logp = t1+t2+t3
    logp_summed = jnp.sum(logp,axis=1,keepdims=True) #Sum the logs to handle multiple dimensions of the action space
    return  logp_summed

def scale_action(action,cliprange):
    """ Scales a [-1,1] action to the desired clip range """
    act = (action+1.)*(cliprange[1]-cliprange[0]) / 2. + cliprange[0]
    return act
