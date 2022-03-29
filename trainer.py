import optax
from rollout import rollout
from better_replay_buffer import ReplayBuffer
from config import DEFAULT_TRAINER_CONFIG
from functools import partial
import gym
import time
from utils import mapped_vectorize
import uuid
from optax._src import combine,clipping

import jax
import jax.numpy as jnp
import time

class Trainer:
  def __init__(self,policy_net_creator,value_net_creator,env_creator,config=DEFAULT_TRAINER_CONFIG,name=None):
    self.env_creator = env_creator

    self.policy_net_creator = policy_net_creator
    self.value_net_creator = value_net_creator

    self.config = config

    self.reset()

    self.name = name
    if name is None:
      self.name = str(uuid.uuid4())

  #--------------------------------------------------------
  #
  #                      DUMP CONFIGS
  #
  #--------------------------------------------------------
  def dump_config(self):
    trainer_config_file = open(os.path.join('save',self.name,'trainer_config.json'),'w')
    json.dump(self.config,trainer_config_file)

  def reset(self):
    #Build environments
    self.train_env = self.env_creator()
    self.eval_env = self.env_creator()

    #Create the replay buffer
    self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"],self.train_env)

    #Get the rng key
    self.params_rng,self.train_rollout_rng,self.test_rollout_rng,self.update_rng = jax.random.split(jax.random.PRNGKey(self.config['seed']),4)

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
      cliprange = (self.train_env.action_space.low,self.train_env.action_space.high)
      self.action_function = partial(agent.select_action,cliprange=cliprange)
      self.explore_action_function = partial(agent.select_action_and_explore,cliprange=cliprange)
      #Builds the network
      self.action_dim = self.train_env.action_space.low.shape[0]
      self.policy_net = self.policy_net_creator(self.action_dim,'continuous')
      self.mode = 'continuous'

    #Jit the network's functions
    self.policy_net_init,self.policy_net_apply = (jax.jit(self.policy_net.init),jax.jit(self.policy_net.apply))

    #Intialize policy network parameters
    self.policy_params = self.policy_net_init(x=self.train_env.observation_space.sample(),rng=self.params_rng)

    #Create the value network
    self.value_net = self.value_net_creator()
    self.value_net_init,self.value_net_apply = (jax.jit(self.value_net.init),jax.jit(self.value_net.apply))

    #Intialize value network parameters
    self.value_params = self.value_net_init(x=self.train_env.observation_space.sample(),rng=self.params_rng)
    

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
    for i in range(nb_steps):
      #-- Training Rollout
      
      #Time the epoch
      t = time.perf_counter()
      
      #Rollout in the train environment to fill the replay buffer
      self.train_rollout_rng,train_rollout_rng = jax.random.split(self.train_rollout_rng)
      data,mean_rew,mean_len = rollout(self.explore_action_function,self.train_env,self.config['training_rollout_length'],self.replay_buffer,self.config['gamma'],self.config['decay'],self.policy_params,self.value_params,self.policy_net_apply,self.value_net_apply,train_rollout_rng,reward_scaling=self.config['reward_scale'])
      
      #Counts the number of timesteps
      nb_stepped += len(data["actions"])
      
      #Fits several time using the replay buffer
      policy_losses = []
      value_losses = []
      new_policy_params = self.policy_params
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample_batch(self.config['train_batch_size'])
        
        self.update_rng,update_rng = jax.random.split(self.update_rng)
        new_policy_params,self.value_params,self.policy_opt_state,self.value_opt_state,policy_loss,value_loss = self.jitted_update(new_policy_params,self.value_params,batch,self.policy_opt_state,self.value_opt_state,self.config['clip_eps'],self.policy_params,update_rng)
        
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
      #Saves the new params
      self.policy_params = new_policy_params

      print("------------------------------")
      if self.mode == 'discrete':
        print('actions mean proba',jnp.mean(mapped_vectorize(self.action_dim)(data["actions"]),axis=0))
      else:
        print('actions mean :',jnp.mean(data["actions"])," std :",jnp.std(data["actions"])," min :",jnp.min(data["actions"])," max:",jnp.max(data["actions"]))
        print("ex :",data["observations"][:10])
        print("ex :",data["actions"][:10,0])

      print('policy loss - mean:',jnp.mean(jnp.array(policy_losses)),'std:',jnp.std(jnp.array(policy_losses)))
      print('value loss - mean:',jnp.mean(jnp.array(value_losses)),'std:',jnp.std(jnp.array(value_losses)))
      
      #Eval Rollout (doesn't put trajectories in the replay buffer)
      self.test_rollout_rng,test_rollout_rng = jax.random.split(self.test_rollout_rng)
      data,mean_rew,mean_len = rollout(self.action_function,self.eval_env,self.config['testing_rollout_length'],self.replay_buffer,self.config['gamma'],self.config['decay'],self.policy_params,self.value_params,self.policy_net_apply,self.value_net_apply,test_rollout_rng,add_buffer=False)

      #Print stats
      print("---------- EPOCH ",i," - nb_steps ",nb_stepped)
      if self.mode == 'discrete':
        print('actions mean proba',jnp.mean(mapped_vectorize(self.action_dim)(data["actions"]),axis=0))
      else:
        print('actions mean :',jnp.mean(data["actions"])," std :",jnp.std(data["actions"])," min :",jnp.min(data["actions"])," max:",jnp.max(data["actions"]))
        print("ex :",data["observations"][:10])
        print("ex :",data["actions"][:10,0])
      print('policy loss - mean:',jnp.mean(jnp.array(policy_losses)),'std:',jnp.std(jnp.array(policy_losses)))
      print('value loss - mean:',jnp.mean(jnp.array(value_losses)),'std:',jnp.std(jnp.array(value_losses)))
      print('mean reward :',mean_rew,' mean len :',mean_len) #Print stats about what's happening during training
      print('time :',time.perf_counter()-t)
