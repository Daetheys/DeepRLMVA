import numpy as np
import jax
import jax.numpy as jnp
import gym
from statistics import mean

from rollout import *
from replay_buffer import ReplayBuffer
from networks import actor_net,value_net
from agent import select_action
from env_wrapper import ActionScalingWrapper

def test_gaes():
    gamma = 0.99
    decay = 0.97
    rewards = np.array([1,1,1,1,1])
    values = np.array([1,1,1,1,1])
    done = np.array([0,0,0,0,0])
    advs = calculate_gaes(rewards,values,done,1,gamma=gamma,decay=decay)

def test_rollout():
    nb_steps = 201
    discount = 0.99
    env_creator = lambda :ActionScalingWrapper(gym.make('Pendulum-v1'))
    env = env_creator()
    rng = jax.random.PRNGKey(42)
    replay_buffer = ReplayBuffer(5,env)
    
    act_net = actor_net(env.action_space.low.shape[0])
    actor_init, actor_apply = act_net
    actor_params = actor_init(rng, jnp.array(env.observation_space.sample()))

    val_net = value_net(env.action_space.low.shape[0])
    value_init, value_apply = val_net
    value_params = value_init(rng, jnp.array(env.observation_space.sample()))

    infos, mean_reward, mean_timestep = rollout(select_action, env, nb_steps, replay_buffer, discount, 0.97, actor_params, value_params,actor_apply, value_apply, rng)
    assert len(infos) == 4
