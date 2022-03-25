import jax
import optax
from rollout import rollout
from agent import update
from replay_buffer import ReplayBuffer
from functools import partial
import gym

import agent

import jax
import jax.numpy as jnp

DEFAULT_CONFIG = {
  "nb_fit_per_epoch":6,
  "train_batch_size":128,
  "training_rollout_length":1000,
  "learning_rate":3e-4,
  "clip_eps":0.3,
  "seed": 42,
  "gamma":0.99
}

jitted_rollout = rollout#jax.jit(rollout)


class Trainer:
  def __init__(self,net,env_creator,config=DEFAULT_CONFIG):
    self.env_creator = env_creator

    self.net = net

    self.config = config

    self.replay_buffer = ReplayBuffer(1e5)

    self.reset()

  def reset(self): #Doesn't reset the network
    self.train_env = self.env_creator()
    self.eval_env = self.env_creator()

    self.rng = jax.random.PRNGKey(self.config['seed'])
    self.params = self.net.init(x=self.train_env.observation_space.sample(),rng=self.rng)

    self.opt = optax.sgd(self.config['learning_rate'])
    self.opt_state = self.opt.init(self.params)

    self.action_function = agent.select_action_continuous
    if isinstance(self.train_env.action_space,gym.spaces.Discrete):
      self.action_function = agent.select_action_discrete

    #jitted functions
    self.jitted_update = jax.jit(partial(update,self.net.apply,self.opt))

  def train(self,nb_steps):
    nb_stepped = 0
    for i in range(nb_steps):
      #Training Rollout
      data,mean_rew,mean_len = jitted_rollout(self.action_function,self.train_env,self.config['training_rollout_length'],self.replay_buffer,self.config['gamma'],self.params,self.net.apply,self.rng) #rng,params,apply
      nb_stepped = len(data["actions"])
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample_batch(self.config['train_batch_size'])
        new_params = self.params
        new_params,self.opt_state = self.jitted_update(new_params,batch,self.opt_state,self.config['clip_eps'],self.params,self.rng)
      self.params = new_params
      #Eval Rollout
      #_,mean_reward,mean_ = jitted_rollout(self.eval_env,self.config['testing_rollout_length'],discount=self.config['gamma'])
      def vectorize(a):
        arr = jnp.zeros(self.train_env.action_space.n)
        arr = arr.at[a].set(1)
        return arr
      mapped_vectorize = jax.vmap(vectorize,0)
      print("---------- TIMESTEP ",i," - nb_steps ",nb_stepped)
      print('actions mean proba',jnp.mean(mapped_vectorize(data["actions"]),axis=0))
      print('mean reward :',mean_rew,' mean len :',mean_len) #Print stats about what's happening during training
