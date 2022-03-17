import jax
import optax
from rollout import rollout
from agent import update
from replay_buffer import ReplayBuffer

DEFAULT_CONFIG = {
  "nb_fit_per_epoch":5,
  "train_batch_size":512,
  "training_rollout_length":1024,
  "learning_rate":0.01
}

jitted_rollout = jax.jit(rollout)
jitted_update = jax.jit(update)
class Trainer:
  def __init__(self,net,env_creator,config=DEFAULT_CONFIG):
    self.env_creator = env_creator

    self.net = net

    self.config = config
    

    self.replay_buffer = ReplayBuffer()

    self.config = config

  def reset(self):
    self.train_env = self.env_creator()
    self.eval_env = self.env_creator()

    self.rng = jax.random.PRNGKey(42)
    self.params = self.net.init(x=self.train_env.observation_space.sample(),rng=self.rng)
    self.opt = optax.adam(self.config['learning_rate'])
    self.opt_state = self.opt.init(self.params)
    self.clip_eps = 0.1

  def train(self,nb_steps):
    for i in range(nb_steps):
      #Training Rollout
      jitted_rollout(self.train_env,self.agent,self.config['training_rollout_length'],replay_buffer=self.replay_buffer)
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample(self.config['train_batch_size'])
        new_params = self.params
        new_params,new_opt_state = jitted_update(self.params,batch,self.opt,self.opt_state,self.clip_eps,self.params)
      self.params = new_params
      #Eval Rollout
      stats = jitted_rollout(self.eval_env,self.agent,self.config['testing_rollout_length'])
      print('stats',stats) #Print stats about what's happening during training
