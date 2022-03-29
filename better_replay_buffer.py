import jax.numpy as jnp
import numpy as np
import collections

Transition = collections.namedtuple("Transition",
                                    field_names=["obs", "act", "rew", "nobs", "logp","discount", "gae"])

class ReplayBuffer:
    def __init__(self,maxlen,env):
        self.maxlen = maxlen

        self.fields = {"obs":np.zeros((maxlen,)+env.observation_space.low.shape),
                       "act":np.zeros((maxlen,)+env.action_space.low.shape),
                       "logp":np.zeros((maxlen,)+(1,)),
                       "nobs":np.zeros((maxlen,)+env.observation_space.low.shape),
                       "rew":np.zeros((maxlen,)+(1,)),
                       "discount":np.zeros((maxlen,)+(1,)),
                       "gae":np.zeros((maxlen,)+(1,))
                       }

        self.cursor = 0
        self.size = 0

    def add(self,*ts):
        ts = Transition(*ts)
        for k in self.fields:
            self.fields[k][self.cursor] = getattr(ts,k)
        self.cursor = (self.cursor+1)%self.maxlen
        self.size = min(self.size+1,self.maxlen)

    def sample_batch(self,batch_size):
        idx_batch = np.random.choice(self.size,batch_size,replace=False)
        d = {}
        for k in self.fields:
            d[k] = self.fields[k][idx_batch]
        return Transition(**d)
