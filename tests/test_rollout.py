import numpy as np
import jax
import jax.numpy as jnp
from statistics import mean

from rollout import *
from replay_buffer import BaseReplayBuffer
from networks import actor_critic_net
import agent 

def test_rollout():
    select_action = agent.select_action_discrete
    env = 0
    nb_steps = 50
    replay_buffer = BaseReplayBuffer(5)
    discount = 0.99
    params= init(rng, jnp.zeros(10))
    net = actor_critic_net(5)
    init, apply = net
    rng = jax.random.PRNGKey(42)
    infos, mean_reward, mean_timestep = rollout(select_action, env, nb_steps, replay_buffer, discount, params, apply, rng)
    assert infos.shape == (4, 50)