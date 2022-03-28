import optax
from rollout import rollout
from replay_buffer import ReplayBuffer
from config import DEFAULT_TRAINER_CONFIG
from functools import partial
import gym
import time
from utils import mapped_vectorize
import uuid

import jax
import jax.numpy as jnp

class Trainer:
  def __init__(self,net_creator,env_creator,config=DEFAULT_TRAINER_CONFIG,name=None):
    self.env_creator = env_creator

    self.net_creator = net_creator

    self.config = config

    self.replay_buffer = ReplayBuffer(1e5)

    self.reset()

    self.name = name
    if name is None:
      self.name = str(uuid.uuid4())

  #--------------------------------------------------------
  #
  #                      DUMP CONFIGS
  #
  #--------------------------------------------------------
      
  def dump_configs(self):
    #Dump trainer config
    self.dump_trainer_config()
    #Dump agent config
    self.dump_agent_config()

  def dump_trainer_config(self):
    trainer_config_file = open(os.path.join('save',self.name,'trainer_config.json'),'w')
    json.dump(self.config,trainer_config_file)

  def dump_agent_config(self):
    agent_config_file = open(os.path.join('save',self.name,'agent_config.json'),'w')
    json.dump(self.agent.config,agent_config_file)

  def reset(self):
    self.train_env = self.env_creator()
    self.eval_env = self.env_creator()

    self.params_rng,self.train_rng,self.test_rng = jax.split(jax.random.PRNGKey(self.config['seed']),3)

    if isinstance(self.train_env.action_space,gym.spaces.Discrete):
      import agent_discrete as agent
      self.action_function = agent.select_action
      self.explore_action_function = agent.select_action
      self.action_dim = self.train_env.action_space.n
      self.net = self.net_creator(self.action_dim,'discrete')
    else:
      import agent_continuous as agent
      self.action_function = agent.select_action
      self.explore_action_function = agent.select_action_and_explore
      self.action_dim = self.train_env.action_space.low.shape[0]
      self.net = self.net_creator(self.action_dim,'continuous')

    self.net_init,self.net_apply = jax.jit(self.net.init),jax.jit(self.net.apply)

    self.params = self.net_init(x=self.train_env.observation_space.sample(),rng=self.params_rng)
    self.opt = optax.sgd(self.config['policy_learning_rate'])
    self.opt_state = self.opt.init(self.params)

    self.jitted_update = partial(agent.update,self.net_apply,self.opt)

  def train(self,nb_steps):
    print('Start Training')
    nb_stepped = 0
    for i in range(nb_steps):
      #Training Rollout
      t = time.perf_counter()
      data,mean_rew,mean_len = rollout(self.explore_action_function,self.train_env,self.config['training_rollout_length'],self.replay_buffer,self.config['gamma'],self.params,self.net_apply,self.train_rng)
      nb_stepped += len(data["actions"])
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample_batch(self.config['train_batch_size'])
        new_params = self.params
        new_params,self.opt_state = self.jitted_update(new_params,batch,self.opt_state,self.config['clip_eps'],self.params,self.rng,self.config['clip_grad'])
      self.params = new_params
      #Eval Rollout
      #data,mean_rew,mean_len = rollout(self.action_function,self.eval_env,self.config['testing_rollout_length'],self.replay_buffer,self.config['gamma'],self.params,self.net_apply,self.test_rng,add_buffer=False)
      print("---------- EPOCH ",i," - nb_steps ",nb_stepped)
      print('actions mean proba',jnp.mean(mapped_vectorize(self.action_dim)(data["actions"]),axis=0))
      print('mean reward :',mean_rew,' mean len :',mean_len) #Print stats about what's happening during training
      print('time :',time.perf_counter()-t)
