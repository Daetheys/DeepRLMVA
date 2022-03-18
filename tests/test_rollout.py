import numpy as np
from statistics import mean

from rollout import *

def test_rollout():
  infos, mean_ep_reward, mean_ep_timestep  = rollout(agent, env, nb_steps, replay_buffer) 