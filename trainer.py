import optax
from rollout import rollout
from replay_buffer import ReplayBuffer
from functools import partial
import gym
import time
from utils import mapped_vectorize
import uuid
from optax._src import combine,clipping

import jax
import jax.numpy as jnp
import time
import haiku as hk
import numpy as np
import os
import json

class Trainer:
  def __init__(self,policy_net_creator,value_net_creator,env_creator,config=None,name=None):
    assert not(config is None) #Please provide a config to the trainer (see preconfig/ folder)
    self.env_creator = env_creator

    self.policy_net_creator = policy_net_creator
    self.value_net_creator = value_net_creator

    self.full_config = config
    self.config = config['trainer']

    self.reset()

    self.name = name
    if name is None:
      self.name = str(uuid.uuid4())

  #--------------------------------------------------------
  #
  #                      DUMP CONFIGS
  #
  #--------------------------------------------------------

  def dump_all(self):
    """ All data about the training """
    path = os.path.join('results',self.name)
    os.makedirs(path)
    self.dump_config()
    self.dump_scores()
    print('---------------------------------------------')
    print('   RESULTS SAVED IN {}    '.format(path))
    print('---------------------------------------------')
  
  def dump_config(self):
    """ Dump the config file which was used to generate the training """
    with open(os.path.join('results',self.name,'config.json'),'w') as f:
      data = json.dumps(self.full_config)
      f.write(data)

  def dump_scores(self):
    """ Dump the scores acquired during the training """
    import xlsxwriter
    file_path = os.path.join('results',self.name,'results.xlsx')
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    print(self.scores)
    for i,data in enumerate(self.scores):
      for j,d in enumerate(data):
        worksheet.write(i,j,d)
    workbook.close()

  def reset(self):
    
    #Build environments
    self.train_env = self.env_creator()
    self.eval_env = self.env_creator()

    #Get the rng key
    self.rng = hk.PRNGSequence(self.config['seed'])
    np.random.seed(self.config['seed'])

    self.train_env.seed(self.config['seed'])
    self.eval_env.seed(2**31-self.config['seed'])
    
    #Create the replay buffer
    self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"],self.train_env)

    #Builds the network depending on the type of environment considered (discrete / continuous action space)
    if isinstance(self.train_env.action_space,gym.spaces.Discrete):
      #Import the right agent
      import agent_discrete as agent
      #Get the right functions
      self.action_function = agent.select_action
      self.explore_action_function = agent.select_action
      #Builds the network
      self.action_dim = self.train_env.action_space.n
      self.policy_net = self.policy_net_creator(self.action_dim,'discrete')
      self.mode = 'discrete'
    else:
      #Import the right agent
      import agent_continuous as agent
      #Get the right functions
      self.action_function = agent.select_action
      self.explore_action_function = agent.select_action_and_explore
      #Builds the network
      self.action_dim = self.train_env.action_space.low.shape[0]
      self.policy_net = self.policy_net_creator(self.action_dim,'continuous')
      self.mode = 'continuous'

    #Create the value network
    self.value_net = self.value_net_creator()
    self.value_net_init,self.value_net_apply = (jax.jit(self.value_net.init),jax.jit(self.value_net.apply))

    #Intialize value network parameters
    self.value_params = self.value_net_init(x=self.train_env.observation_space.sample(),rng=next(self.rng))

    #Jit the network's functions
    self.policy_net_init,self.policy_net_apply = (jax.jit(self.policy_net.init),jax.jit(self.policy_net.apply))

    #Intialize policy network parameters
    self.policy_params = self.policy_net_init(x=self.train_env.observation_space.sample(),rng=next(self.rng))

    #Builds the policy optimizer
    if self.config['policy_optimizer'] == 'sgd':
      self.policy_opt = optax.sgd(self.config['policy_learning_rate'])
    elif self.config['policy_optimizer'] == 'adam':
      self.policy_opt = optax.adam(self.config['policy_learning_rate'])
    #Add gradient clipping
    self.policy_opt = combine.chain(
      clipping.clip_by_global_norm(self.config["clip_grad"]),
      self.policy_opt)
    #Init state
    self.policy_opt_state = self.policy_opt.init(self.policy_params)

    #Builds the value optimizer
    if self.config['value_optimizer'] == 'sgd':
      self.value_opt = optax.sgd(self.config['value_learning_rate'])
    elif self.config['value_optimizer'] == 'adam':
      self.value_opt = optax.adam(self.config['value_learning_rate'])
    #Add gradient clipping
    self.value_opt = combine.chain(
      clipping.clip_by_global_norm(self.config["clip_grad"]),
      self.value_opt)
    #Init state
    self.value_opt_state = self.value_opt.init(self.value_params)

    #Jit the right update function
    self.jitted_update = jax.jit(partial(agent.update,self.policy_net_apply,self.value_net_apply,self.policy_opt,self.value_opt))


  def train(self,nb_steps):
    print('Start Training')
    #Counts the current timestep
    nb_stepped = 0
    epoch = 0
    self.scores = [('timestep','mean_reward')]
    while nb_stepped < nb_steps:
      #-- Training Rollout
      epoch += 1
      
      #Time the epoch
      t = time.perf_counter()
      
      #Rollout in the train environment to fill the replay buffer
      train_data,_,_ = rollout(self.explore_action_function,self.train_env,self.config['training_rollout_length'],self.replay_buffer,self.config['gamma'],self.config['decay'],self.policy_params,self.value_params,self.policy_net_apply,self.value_net_apply,self.rng,reward_scaling=self.config['reward_scale'],mask_done=self.config['mask_done'])
      
      #Counts the number of timesteps
      nb_stepped += len(train_data["actions"])
      
      #Fits several time using the replay buffer
      policy_losses = []
      value_losses = []
      policy_gradients = []
      value_gradients = []
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample_batch(self.config['train_batch_size'])

        self.policy_params,self.value_params,self.policy_opt_state,self.value_opt_state,policy_loss,value_loss,policy_grads,value_grads = self.jitted_update(self.policy_params,self.value_params,batch,self.policy_opt_state,self.value_opt_state,self.config['clip_eps'],self.config["kl_coeff"],self.config["entropy_coeff"])
          
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        policy_gradients.append(policy_grads)
        value_gradients.append(value_grads)
      
      #Eval Rollout (doesn't put trajectories in the replay buffer)
      test_data,mean_rew,mean_len = rollout(self.action_function,self.eval_env,self.config['testing_rollout_length'],self.replay_buffer,self.config['gamma'],self.config['decay'],self.policy_params,self.value_params,self.policy_net_apply,self.value_net_apply,self.rng,add_buffer=False)
      
      #Save stats
      self.scores.append((nb_stepped,mean_rew))

      #Print stats
      print("---------- EPOCH ",epoch," - nb_steps ",nb_stepped,' - mean reward ',mean_rew,' - mean length ',mean_len)
      if self.mode == 'discrete':
        print('actions mean proba',jnp.mean(mapped_vectorize(self.action_dim)(train_data["actions"]),axis=0))
      else:
        print('TRAIN actions mean :',jnp.mean(train_data["actions"])," std :",jnp.std(train_data["actions"])," min :",jnp.min(train_data["actions"])," max:",jnp.max(train_data["actions"]))
        print('TEST actions mean :',jnp.mean(test_data["actions"])," std :",jnp.std(test_data["actions"])," min :",jnp.min(test_data["actions"])," max:",jnp.max(test_data["actions"]))
      print('policy loss - mean:',jnp.mean(jnp.array(policy_losses)),'std:',jnp.std(jnp.array(policy_losses)))
      print('|-> policy gradients - mean:',jnp.mean(jnp.array(policy_gradients)),'std:',jnp.std(jnp.array(policy_gradients)))
      print('value loss - mean:',jnp.mean(jnp.array(value_losses)),'std:',jnp.std(jnp.array(value_losses)))
      print('|-> value gradients - mean:',jnp.mean(jnp.array(value_gradients)),'std:',jnp.std(jnp.array(value_gradients)))
      print('time :',time.perf_counter()-t)

    #End of training
    self.dump_all()
