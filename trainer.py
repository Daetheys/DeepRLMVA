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
import haiku as hk
import numpy as np

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
      cliprange = (self.train_env.action_space.low,self.train_env.action_space.high)
      self.action_function = partial(agent.select_action,cliprange=cliprange)
      self.explore_action_function = partial(agent.select_action_and_explore,cliprange=cliprange)
      #Builds the network
      self.action_dim = self.train_env.action_space.low.shape[0]
      self.policy_net = self.policy_net_creator(self.action_dim,'continuous')
      self.mode = 'continuous'

    #Create the value network
    self.value_net = self.value_net_creator()
    #self.value_net_init,self.value_net_apply = (jax.jit(self.value_net.init),jax.jit(self.value_net.apply))
    self.value_net_init,self.value_net_apply = (self.value_net.init,self.value_net.apply)

    #Intialize value network parameters
    self.value_params = self.value_net_init(x=self.train_env.observation_space.sample(),rng=next(self.rng))

    #Jit the network's functions
    #self.policy_net_init,self.policy_net_apply = (jax.jit(self.policy_net.init),jax.jit(self.policy_net.apply))
    self.policy_net_init,self.policy_net_apply = (self.policy_net.init,self.policy_net.apply)

    #Intialize policy network parameters
    self.policy_params = self.policy_net_init(x=self.train_env.observation_space.sample(),rng=next(self.rng))

    """print("Checkpoint VALUE NETWORK")
    x = jnp.ones((1,3))
    out = self.value_net_apply(self.value_params,x)
    print(out)

    print("Checkpoint ACTOR NETWORK")
    x = jnp.ones((1,3))
    out = self.policy_net_apply(self.policy_params,x)
    print(out)
    assert False"""
    

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
      train_data,mean_rew,mean_len = rollout(self.explore_action_function,self.train_env,self.config['training_rollout_length'],self.replay_buffer,self.config['gamma'],self.config['decay'],self.policy_params,self.value_params,self.policy_net_apply,self.value_net_apply,self.rng,reward_scaling=self.config['reward_scale'])
      
      #Counts the number of timesteps
      nb_stepped += len(train_data["actions"])

      """print('CHECKPOINT REPLAY BUFFER')
      for s,a,r,d,logp,ns,g,t in zip(
          self.replay_buffer.fields['obs'][:10],
          self.replay_buffer.fields['act'][:10],
          self.replay_buffer.fields['rew'][:10],
          self.replay_buffer.fields['discount'][:10],
          self.replay_buffer.fields['logp'][:10],
           self.replay_buffer.fields['nobs'][:10],
          self.replay_buffer.fields['gae'][:10],
          self.replay_buffer.fields['gae'][:10]+self.replay_buffer.fields['values'][:10]):
        print(s,a,r,d,logp,ns,g,t)
        
      #assert False"""
      
      #Fits several time using the replay buffer
      policy_losses = []
      value_losses = []
      for j in range(self.config['nb_fit_per_epoch']):
        self.replay_buffer.shuffle()
        for k in range(0,self.replay_buffer.size,self.config['train_batch_size']):
          batch = self.replay_buffer.sample_batch(k,self.config['train_batch_size'])

          """print("CHECKPOINT UPDATE")
          print(batch.obs.shape)
          print(batch.obs[:10])
          print(batch.act[:10])
          print(batch.logp[:10])
          gae = batch.gae
          gae = (gae-gae.mean())/(gae.std()+1e-8)
          print(gae[:10])
          print(batch.gae[:10]+batch.values[:10])
          #assert False"""

          self.policy_params,self.value_params,self.policy_opt_state,self.value_opt_state,policy_loss,value_loss = self.jitted_update(self.policy_params,self.value_params,batch,self.policy_opt_state,self.value_opt_state,self.config['clip_eps'],self.config["kl_coeff"],self.config["entropy_coeff"])

          #print('CHECKPOINT LOSS')
          #print(value_loss,policy_loss)
          #assert False
          
          """(a,m,s,logpis,loss) = to_debug
          print("---",loss)
          print(m.min(),m.max(),s.min(),s.max())
          print(a.min(),a.max(),logpis.min(),logpis.max())
          assert not(jnp.isnan(loss))"""
          """print('---')
          for k in self.value_params:
          for k2 in self.value_params[k]:
            print(k,k2,jnp.mean(self.value_params[k][k2]))
          if j>10:
          assert False"""
          
          policy_losses.append(policy_loss)
          value_losses.append(value_loss)

      #Empty the replay buffer (not necessary - just to be sure)
      #self.replay_buffer.empty()
      
      #Eval Rollout (doesn't put trajectories in the replay buffer)
      test_data,mean_rew,mean_len = rollout(self.action_function,self.eval_env,self.config['testing_rollout_length'],self.replay_buffer,self.config['gamma'],self.config['decay'],self.policy_params,self.value_params,self.policy_net_apply,self.value_net_apply,self.rng,add_buffer=False)

      #Print stats
      print("---------- EPOCH ",i," - nb_steps ",nb_stepped)
      if self.mode == 'discrete':
        print('actions mean proba',jnp.mean(mapped_vectorize(self.action_dim)(train_data["actions"]),axis=0))
      else:
        print('TRAIN actions mean :',jnp.mean(train_data["actions"])," std :",jnp.std(train_data["actions"])," min :",jnp.min(train_data["actions"])," max:",jnp.max(train_data["actions"]))
        print('TEST actions mean :',jnp.mean(test_data["actions"])," std :",jnp.std(test_data["actions"])," min :",jnp.min(test_data["actions"])," max:",jnp.max(test_data["actions"]))
      print('policy loss - mean:',jnp.mean(jnp.array(policy_losses)),'std:',jnp.std(jnp.array(policy_losses)))
      print('value loss - mean:',jnp.mean(jnp.array(value_losses)),'std:',jnp.std(jnp.array(value_losses)))
      print('mean reward :',mean_rew,' mean len :',mean_len) #Print stats about what's happening during training
      print('time :',time.perf_counter()-t)
