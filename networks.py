import haiku as hk
import jax
import jax.numpy as jnp

def actor_net(out_dim,layer_size=256):
    def _wrap(x):
        init_hidden = hk.initializers.Orthogonal(scale=1.)
        init_out = hk.initializers.Orthogonal(scale=0.01)
        mu = hk.Sequential([hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                            hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                            hk.Linear(out_dim,w_init=init_out)])(x)
        zinit = hk.initializers.Constant(0.)
        logstd = hk.get_parameter('logstd', (1,out_dim),init=zinit) #Learning the std directly diverges so we used the trick to learn a constant vector instead
        return mu,logstd
    return hk.without_apply_rng(hk.transform(_wrap))

def value_net(layer_size=256):
    def _wrap(x):
        init_hidden = hk.initializers.Orthogonal(scale=1.)
        init_out = hk.initializers.Orthogonal(scale=1.)
        val = hk.Sequential([hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                            hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                            hk.Linear(1,w_init=init_out)])(x)
        return val
    return hk.without_apply_rng(hk.transform(_wrap))
