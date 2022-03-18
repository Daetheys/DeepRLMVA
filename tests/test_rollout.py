import numpy as np
from statistics import mean

from rollout import *

def test_rollout():
    agent = 0
    env = 0
    nb_steps = 0
    replay_buffer = 0
    discount = 0.99
    params, apply = 0,0
    rng = jax.random.PRNGKey(42)
    infos, mean_reward, mean_timestep = rollout(agent, env, nb_steps, replay_buffer, discount, params, apply, rng)