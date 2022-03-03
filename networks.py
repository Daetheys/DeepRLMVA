import haiku as hk
import jax

#---------------------------------------------------------------------
#                              Builders
#---------------------------------------------------------------------

def build_network(instr):
    """ Generic builder for neural networks """
    l = []
    for cl,args in range(len(instr)):
        l.append( cl(*args) )
    return hk.Sequential(l)

def build_mlp(hidden_dims,activation):
    """ MLP builder (stacks Linear layers) """
    l = []
    for i in range(len(hidden_dims)):
        l.append((hk.Linear,(hidden_dims[i])))
        l.append((lambda : activation,None))
    return hk.Sequential(l)

# -> Add Conv builders ? Attention builders ?

#---------------------------------------------------------------------
#                              Networks
#---------------------------------------------------------------------

def simple_net(out_dim):
    return build_mlp([256,256,out_dim],jax.nn.relu)