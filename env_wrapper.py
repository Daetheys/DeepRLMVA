import numpy as np
import gym
class EnvWrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.env = env
    def step(self,a):
        return self.env.step(np.array(a))