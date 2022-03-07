from logger import logger
import uuid

DEFAULT_TRAINER_CONFIG = {
  "nb_fit_per_epoch":5,
  "train_batch_size":512,
  "training_rollout_length":1024,
  ""
}

class Trainer:
  def __init__(self,agent,env_creator,config=DEFAULT_CONFIG,name=None):
    self.agent = agent

    self.train_env = env_creator()
    self.eval_env = env_creator()

    #self.replay_buffer = ReplayBuffer()

    self.config = config

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

  #--------------------------------------------------------
  #
  #                         TRAIN
  #
  #--------------------------------------------------------
    
  def train(self,nb_steps):
    #Saves the config
    self.dump_configs()
    for i in range(nb_steps):
      #Training Rollout
      rollout(self.train_env,self.agent,self.config['training_rollout_length'],replay_buffer=self.replay_buffer)
      for j in range(self.config['nb_fit_per_epoch']):
        batch = self.replay_buffer.sample(self.config['train_batch_size'])
        self.agent.fit(batch)
      #Eval Rollout
      rollout(self.eval_env,self.agent,self.config['testing_rollout_length'])
      logger.info("")
