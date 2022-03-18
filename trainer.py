DEFAULT_CONFIG = {
  "nb_fit_per_epoch":5,
  "train_batch_size":512,
  "training_rollout_length":1024,
  ""
}

class Trainer:
  def __init__(self,agent,env_creator,config=DEFAULT_CONFIG):
    self.agent = agent

    self.train_env = env_creator()
    self.eval_env = env_creator()

    #self.replay_buffer = ReplayBuffer()

    self.config = config

  def train(self,nb_steps):
    for i in range(nb_steps):
      #Training Rollout
      #rollout(self.train_env,self.agent,self.config['training_rollout_length'],replay_buffer=self.replay_buffer)
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample(self.config['train_batch_size'])
        self.agent.fit(batch)
      #Eval Rollout
      #rollout(self.eval_env,self.agent,self.config['testing_rollout_length'])
      #print('stats') #Print stats about what's happening during training
