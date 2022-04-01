from utils import *
import jax.numpy as jnp

def test_compute_logprob():
    actions = jnp.array([[0.2,0.3,0.4],[0.7,0.6,0.5]])
    logstd = jnp.array([[-1.,-2.,-3.],[-4.,-5.,-6.]])
    normal = jnp.array([[0.5,0.6,0.7],[0.8,0.9,1.0]])
    logp = compute_logprob_tanh(actions,logstd,normal)
    print(logp)
    assert jnp.all(jnp.isclose(logp,jnp.array([[3.0026672],[12.425493]])))


def test_scale_action():
    action = 0
    cliprange = [-1,1]
    assert scale_action(action,cliprange) == 0
    cliprange = [0,2]
    assert scale_action(action,cliprange) == 1
    action = 1
    assert scale_action(action,cliprange) == 2
