import jax
import optax
from rollout import rollout
from agent import update
from replay_buffer import ReplayBuffer

import agent

DEFAULT_CONFIG = {
  "nb_fit_per_epoch":5,
  "train_batch_size":512,
  "training_rollout_length":1024,
  "learning_rate":0.01,
  "clip_eps":0.1,
  "seed":42,
  "gamma":0.99
}

jitted_rollout = rollout#jax.jit(rollout)
jitted_update = jax.jit(update)

class Trainer:
  def __init__(self,net,env_creator,config=DEFAULT_CONFIG):
    self.env_creator = env_creator

    self.net = net

    self.config = config

    self.replay_buffer = ReplayBuffer(1e5)

    self.reset()

  def reset(self):
    self.train_env = self.env_creator()
    self.eval_env = self.env_creator()

    self.rng = jax.random.PRNGKey(self.config['seed'])
    self.params = self.net.init(x=self.train_env.observation_space.sample(),rng=self.rng)
    self.opt = optax.adam(self.config['learning_rate'])
    self.opt_state = self.opt.init(self.params)

  def train(self,nb_steps):
    for i in range(nb_steps):
      #Training Rollout
      _,mean_rew,mean_len = jitted_rollout(agent.select_action_discrete,self.train_env,self.config['training_rollout_length'],self.replay_buffer,self.config['gamma'],self.params,self.net.apply,self.rng) #rng,params,apply
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample_batch(self.config['train_batch_size'])
        new_params = self.params
        new_params,new_opt_state = jitted_update(self.params,batch,self.opt,self.opt_state,self.config['clip_eps'],self.params)
      self.params = new_params
      #Eval Rollout
      #_,mean_reward,mean_ = jitted_rollout(self.eval_env,self.config['testing_rollout_length'],discount=self.config['gamma'])
      print('stats',mean_rew,mean_len) #Print stats about what's happening during training
