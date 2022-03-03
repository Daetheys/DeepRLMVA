import pytest
import haiku as hk
import jax.numpy as jnp
import jax

from networks import *

def run_net(net,inp_shape):
    init,fwd = hk.transform(net)
    rng = jax.random.PRNGKey(0)
    x = jnp.zeros(inp_shape)
    params = init(rng=rng,x=x)
    out = fwd(params=params,x=x,rng=rng)
    return out

def test_build_network():
    #Test empty
    net = build_network([])
    out = run_net(net,(10,))
    assert out.shape == (10,)
    out = run_net(net,(3,))
    assert out.shape == (3,)
    #Test standard
    net = build_network([(hk.Linear,64),(lambda : jax.nn.relu,None),(hk.Linear(4))])
    out = run_net(net,(10,))
    assert out.shape == (4,)
    out = run_net(net,(21,))
    assert out.shape == (4,)

def test_mlp_network():
    #Test empty
    net = build_mlp([],None)
    out = run_net(net,(10,))
    assert out.shape == (10,)
    out = run_net(net,(3,))
    assert out.shape == (3,)
    #Test standard
    net = build_mlp([64,64,2],jax.nn.relu)
    out = run_net(net,(10,))
    assert out.shape == (2,)
    out = run_net(net,(21,))
    assert out.shape == (2,)

def test_simple_net():
    #Test standard
    net = simple_net(5)
    out = run_net(net,(10,))
    assert out.shape == (5,)
    out = run_net(net,(21,))
    assert out.shape == (5,)