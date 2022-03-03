import haiku as hk
import jax

#---------------------------------------------------------------------
#                              Builders
#---------------------------------------------------------------------

def build_network(instr):
    """ Generic builder for neural networks"""
    l = []
    for cl,args in range(len(instr)):
        l.append( cl(*args) )
    return hk.Sequential(l)

def mlp_network(hidden_dims,activation):
    """ MLP builder """
    l = []
    for i in range(len(hidden_dims)):
        l.append((hk.Linear,(hidden_dims[i])))
        l.append(lambda : activation)
    return hk.Sequential(l)

# -> Add Conv builders ? Attention builders ?

#---------------------------------------------------------------------
#                              Networks
#---------------------------------------------------------------------

def simple_net(inp_dim,out_dim):
    return mlp_network([inp_dim,256,256,out_dim],jax.nn.relu)