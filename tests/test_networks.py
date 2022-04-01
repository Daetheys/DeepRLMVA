import pytest
import haiku as hk
import jax.numpy as jnp
import jax

from networks import *

def run_net(net,inp_shape):
    init,fwd = net
    rng = jax.random.PRNGKey(42)
    x = jnp.zeros(inp_shape)
    params = init(rng,x)
    out = fwd(params=params,x=x)
    return out

def test_actor_net():
    net = actor_net(5)
    mu,logstd = run_net(net,(45,3))
    assert mu.shape == (45,5)
    assert logstd.shape == (45,5)
    mu,logstd = run_net(net,(23,15))
    assert mu.shape == (23,5)
    assert logstd.shape == (23,5)

def test_critic_net():
    net = value_net(5)
    val = run_net(net,(45,3))
    assert val.shape == (45,1)
    val = run_net(net,(23,15))
    assert val.shape == (23,1)
