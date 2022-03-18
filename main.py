from trainer import Trainer,DEFAULT_CONFIG
import gym
from networks import actor_critic_net

env_creator = lambda : gym.make('CartPole-v1')
env = env_creator()
net = actor_critic_net(env.action_space.n)

trainer = Trainer(net,env_creator,config=DEFAULT_CONFIG)

trainer.train(10)