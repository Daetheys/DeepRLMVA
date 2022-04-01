import haiku as hk
import jax
import jax.numpy as jnp

#---------------------------------------------------------------------
#                              Builders
#---------------------------------------------------------------------

def build_network(instr):
    """ Generic builder for neural networks """
    def _wrap(x):
        l = []
        print(instr)
        for cl,args in instr:
            l.append( cl(*args) )
        model = hk.Sequential(l)
        return model(x)
    return hk.transform(_wrap)

def build_mlp(hidden_dims,activation):
    """ MLP builder (stacks Linear layers) """
    l = []
    for i in range(len(hidden_dims)):
        l.append((hk.Linear,([hidden_dims[i]])))
        l.append((lambda x : activation,[None]))
    return build_network(l)

# -> Add Conv builders ? Attention builders ?

#---------------------------------------------------------------------
#                              Networks
#---------------------------------------------------------------------

def simple_net(out_dim):
    return build_mlp([256,256,out_dim],jax.nn.relu)


def actor_net(out_dim,mode='discrete',layer_size=256):
    def _wrap(x):
        init_hidden = hk.initializers.Orthogonal(scale=1.)
        init_out = hk.initializers.Orthogonal(scale=0.01)
        mu = hk.Sequential([hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                            hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                            hk.Linear(out_dim,w_init=init_out)])(x)
        logstd = hk.get_parameter('log_std', (1,out_dim),init=jnp.zeros)
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



def actor_net2(out_dim,mode='discrete',layer_size=64,min_logstd=-20,max_logstd=2):
    #Define wrapper
    if mode == 'discrete':
        def _wrap(x):
            #init = hk.initializers.RandomUniform(-3e-3,3e-3)
            init = hk.initializers.Constant(0.)
            policy_net = hk.Sequential([hk.Linear(layer_size),jax.nn.relu,
                                        hk.Linear(layer_size),jax.nn.relu,
                                        hk.Linear(out_dim,w_init=init,b_init=init),jax.nn.softmax])
            return policy_net(x)
    else:
        def _wrap(x):
            #init = hk.initializers.RandomUniform(-3e-3,3e-3)
            init_hidden = hk.initializers.Orthogonal(scale=1.)
            init_out = hk.initializers.Orthogonal(scale=0.001)
            policy_net = hk.Sequential([hk.Linear(layer_size,w_init=init_hidden),jnp.tanh,
                                        hk.Linear(layer_size,w_init=init_out),jnp.tanh])
            mean_head = hk.Sequential([hk.Linear(out_dim)])
            logstd_head = hk.Sequential([hk.Linear(out_dim,w_init=init,b_init=init)])
            policy_base = policy_net(x)
            mu = mean_head(policy_base)
            logstd = logstd_head(policy_base)
            #scaled_logstd = min_logstd+1/2*(max_logstd-min_logstd)*(jnp.tanh(logstd)+1)
            #clipped_logstd = jnp.clip(logstd,min_logstd,max_logstd)
            #std = jnp.exp(scaled_logstd)
            return mu,logstd

    return hk.without_apply_rng(hk.transform(_wrap))

def value_net2(layer_size=64):
    def _wrap(x):
        init = hk.initializers.RandomUniform(-3e-3,3e-3)
        #init = hk.initializers.Constant(1.)
        value_net = hk.Sequential([hk.Linear(layer_size),jax.nn.relu,
                                   hk.Linear(layer_size),jax.nn.relu,
                                   hk.Linear(1,w_init=init,b_init=init)])
        return value_net(x)
    return hk.without_apply_rng(hk.transform(_wrap))







































def actor_critic_net(out_dim,mode='discrete',layer_size=256):
    #Define wrapper
    if mode == 'discrete':
        def _wrap(x):
            policy_net = hk.Sequential([hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(out_dim),jax.nn.softmax])
            value_net = hk.Sequential([hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(1)])
            return policy_net(x),value_net(x)
    else:
        def _wrap(x):
            policy_net = hk.Sequential([hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(layer_size),jax.nn.tanh])
            mean_head = hk.Sequential([hk.Linear(out_dim)])
            logstd_head = hk.Sequential([hk.Linear(out_dim)])
            value_net = hk.Sequential([hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(layer_size),jax.nn.tanh,
                                        hk.Linear(1)])
            policy_base = policy_net(x)
            mu = mean_head(policy_base)
            logstd = logstd_head(policy_base)
            logstd = jnp.clip(logstd,-10,2)#*0+0
            std = jnp.exp(logstd)
            return (mu,std),value_net(x)

    return hk.transform(_wrap)

def actor_critic_net_shared(out_dim):
    def _wrap(x):
        base_net = hk.Sequential([  hk.Linear(256),jax.nn.tanh,
                                    hk.Linear(256),jax.nn.tanh,])
        policy_head = hk.Sequential([hk.Linear(out_dim),jax.nn.softmax])
        value_head = hk.Sequential([hk.Linear(1)])
        base = base_net(x)
        return policy_head(base),value_head(base)
    return hk.transform(_wrap)
    
def continuous_actor_critic_net(out_dim):
    def _wrap(x):
        base_net = hk.Sequential([hk.Linear(256),jax.nn.relu])
        policy_head = hk.Sequential([hk.Linear(256),jax.nn.relu])
        mean_policy_head = hk.Sequential([hk.Linear(out_dim)])
        logstd_policy_head = hk.Sequential([hk.Linear(out_dim)])
        value_head = hk.Sequential([hk.Linear(1)])
        base = base_net(x)
        policy_base = policy_head(base)
        return (mean_policy_head(policy_base),logstd_policy_head(policy_base)),value_head(base)
    return hk.transform(_wrap)
