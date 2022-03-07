from trainer import Trainer
from agent import PPOAgent
import gym

env = gym.make('Reacher-v2')
agent = PPOAgent()

trainer = Trainer(env,agent)
trainer.train(1000)
