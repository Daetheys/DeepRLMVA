import jax.numpy as jnp
import numpy as np
import collections
import gym

Transition = collections.namedtuple("Transition",
                                    field_names=["obs", "act", "rew", "nobs", "logp","discount", "gae","return_"])

class ReplayBuffer:
    def __init__(self,maxlen,env):
        self.maxlen = maxlen
        self.env = env

        self.empty()

    def empty(self):
        """ Reset the replay buffer """
        maxlen = self.maxlen
        env = self.env
        
        action_dim = env.action_space.low.shape if isinstance(env.action_space,gym.spaces.Box) else (env.action_space.n,)

        #Reset all fields
        self.fields = {"obs":np.zeros((maxlen,)+env.observation_space.low.shape,dtype=np.float32),
                       "act":np.zeros((maxlen,)+action_dim,dtype=np.float32),
                       "logp":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "nobs":np.zeros((maxlen,)+env.observation_space.low.shape,dtype=np.float32),
                       "rew":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "discount":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "gae":np.zeros((maxlen,)+(1,),dtype=np.float32),
                       "return_":np.zeros((maxlen,)+(1,),dtype=np.float32)
                       }

        #Reset the cursor position used to add timesteps and the size of filled area in the buffer
        self.cursor = 0
        self.size = 0

    def add(self,*ts):
        """ Adds a new timestep in the replay buffer """
        ts = Transition(*ts)
        for k in self.fields:
            self.fields[k][self.cursor] = getattr(ts,k)
        self.cursor = (self.cursor+1)%self.maxlen
        self.size = min(self.size+1,self.maxlen)

    def sample_batch(self,batch_size):
        """ Samples a batch of given size uniformly from the replay buffer (doesn't remove sampled timesteps) """
        idx_batch = np.random.choice(self.size,batch_size,replace=False)
        d = {}
        for k in self.fields:
            d[k] = self.fields[k][idx_batch]
        return Transition(**d)
