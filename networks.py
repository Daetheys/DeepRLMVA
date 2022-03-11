import haiku as hk
import jax

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