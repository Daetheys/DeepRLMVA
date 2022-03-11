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
    out = fwd(params=params,x=x,rng=rng)
    return out

def test_build_network():
    #Test empty
    net = build_network([])
    out = run_net(net,(15,10))
    assert out.shape == (15,10)
    out = run_net(net,(35,3))
    assert out.shape == (35,3)
    #Test standard
    net = build_network([(hk.Linear,[64]),(lambda x: jax.nn.relu,[None]),(hk.Linear,[4])])
    out = run_net(net,(40,10))
    assert out.shape == (40,4)
    out = run_net(net,(0,21))
    assert out.shape == (0,4)

def test_mlp_network():
    #Test empty
    net = build_mlp([],None)
    out = run_net(net,(17,10))
    assert out.shape == (17,10)
    out = run_net(net,(18,3))
    assert out.shape == (18,3)
    #Test standard
    net = build_mlp([64,64,2],jax.nn.relu)
    out = run_net(net,(0,10))
    assert out.shape == (0,2)
    out = run_net(net,(21,))
    assert out.shape == (2,)

def test_simple_net():
    #Test standard
    net = simple_net(5)
    out = run_net(net,(50,10))
    assert out.shape == (50,5)
    out = run_net(net,(20,21))
    assert out.shape == (20,5)