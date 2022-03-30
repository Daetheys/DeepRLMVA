import jax.numpy as jnp
import numpy as np
import collections
import gym

Transition = collections.namedtuple("Transition",
                                    field_names=["obs", "act", "rew", "nobs", "logp","discount", "gae","values"])

class ReplayBuffer:
    def __init__(self,maxlen,env):
        self.maxlen = maxlen
        self.env = env

        self.empty()

    def empty(self):
        maxlen = self.maxlen
        env = self.env
        action_dim = env.action_space.low.shape if isinstance(env.action_space,gym.spaces.Box) else (env.action_space.n,)
        self.fields = {"obs":np.zeros((maxlen,)+env.observation_space.low.shape,dtype=np.float32),
                       "act":np.zeros((maxlen,)+action_dim,dtype=np.float32),
                       "logp":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "nobs":np.zeros((maxlen,)+env.observation_space.low.shape,dtype=np.float32),
                       "rew":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "discount":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "gae":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "values":np.zeros((maxlen,)+(1,),dtype=np.float32)
                       }

        self.cursor = 0
        self.size = 0

        self.replay_cursor = 0
        self.idxs = np.arange(self.maxlen)

    def shuffle(self):
        np.random.shuffle(self.idxs)

    def add(self,*ts):
        ts = Transition(*ts)
        for k in self.fields:
            self.fields[k][self.cursor] = getattr(ts,k)
        self.cursor = (self.cursor+1)%self.maxlen
        self.size = min(self.size+1,self.maxlen)

    def sample_batch2(self,batch_size):
        idx_batch = np.random.choice(self.size,batch_size,replace=False)
        d = {}
        for k in self.fields:
            d[k] = self.fields[k][idx_batch]
        return Transition(**d)

    def sample_batch(self,index,batch_size):
        self.replay_cursor = index
        d = {}
        for k in self.fields:
            d[k] = self.fields[k][self.idxs[self.replay_cursor:self.replay_cursor+batch_size]]
        self.replay_cursor = (self.replay_cursor+batch_size)%self.size
        return Transition(**d)
