import numpy as np
import gym
import multiprocessing as mp
import tree
from utils import scale_action
import jax.numpy as jnp

class ActionScalingWrapper(gym.Wrapper):
    """ Wrap the environment to normalize actions """
    def __init__(self,env):
        super().__init__(env)
        self.env = env
        self.clip_range = self.action_space.low,self.action_space.high
        self._max_episode_steps = env._max_episode_steps
    def step(self,a):
        a = np.array(a)
        a = jnp.clip(scale_action(a,self.clip_range),self.clip_range[0],self.clip_range[1]) #Scale from [-1,1] to the clip range
        return self.env.step(a)

#----------------------------------------------------------------------
#
#             Multi Threading - Not used in the implementation
#
#----------------------------------------------------------------------

def run_env(env,pipe):
    """ Run function for parallel workers """
    while True:
        msg = pipe.recv()
        if msg[0] == 1:
            out = env.reset()
        elif msg[0] == 0:
            out = env.step(msg[1])
        else:
            break
        pipe.send(out)

class ThreadedWrapper(gym.Wrapper):
    """ Threaded Environment Wrapper : Executes the environment in a distant thread """
    def __init__(self,env):
        super().__init__(env)
        self.env = env
        self.pipe,pipe = mp.Pipe(duplex=True)
        self.process = mp.Process(target=run_env,args=(env,pipe),daemon=True)
        self.start()

    def start(self):
        self.process.start()

    def stop(self):
        self.process.terminate()

    def reset(self):
        self.pipe.send((1,))
        return self.pipe.recv()

    def step(self,a):
        self.pipe.send((0,a))
        return self.pipe.recv()

    def close(self):
        self.stop()

    def kill(self):
        self.process.kill()

class ParallelEnv:
    """ Environment that will cluster several environments into a single one """
    def __init__(self,env_creator,nb_env):
        self.envs = [env_creator() for i in range(nb_env)]
        self.obs = np.array([e.observation_space.sample() for e in self.envs])
        self.done = np.array([True for i in range(nb_env)])

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """ Reset the environments which needs it """
        self.obs = np.array([e.reset() if d else o for e,d,o in zip(self.envs,self.done,self.obs)])
        return self.obs

    def force_reset(self):
        """ Force the reset of all environments """
        self.obs = np.array([e.reset() for e in self.envs])
        return self.obs

    def step(self,a):
        """ Steps in each environment """
        out = [e.step(act) for e,act in zip(self.envs,a)]
        out = tree.map_structure(lambda *x : np.stack(x,axis=0),*out)
        self.obs = out[0]
        self.done = out[2]
        return out

    def close(self):
        """ Close all environments """
        for e in self.envs:
            e.close()
