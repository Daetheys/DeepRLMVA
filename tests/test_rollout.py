import numpy as np
import jax
import jax.numpy as jnp
import gym
from statistics import mean

from rollout import *
from replay_buffer import BaseReplayBuffer
from networks import actor_critic_net
from agent import select_action_discrete

def test_rollout():
    select_action = select_action_discrete
    nb_steps = 50
    replay_buffer = BaseReplayBuffer(5)
    discount = 0.99
    env_creator = lambda :EnvWrapper(gym.make('CartPole-v1'))
    env = env_creator()
    net = actor_critic_net(env.action_space.n)
    init, apply = net
    rng = jax.random.PRNGKey(42)
    params= init(rng, jnp.zeros(4))
    rng = jax.random.PRNGKey(42)

    infos, mean_reward, mean_timestep = rollout(select_action, env, nb_steps, replay_buffer, discount, params, apply, rng)
    assert len(infos) == 4
