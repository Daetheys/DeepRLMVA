from trainer import Trainer,DEFAULT_CONFIG
import gym
from networks import actor_critic_net
from env_wrapper import *


if __name__ == '__main__':
    #Build creators
    def env_creator():
        threaded_env_creator = lambda : ThreadedWrapper(JaxWrapper(gym.make('CartPole-v1')))
        return threaded_env_creator()
        #threaded_env_creator = lambda : ThreadedWrapper(JaxWrapper(gym.make('Pendulum-v1')))
        return ParallelEnv(threaded_env_creator,1)
    net_creator = actor_critic_net

    #Build Training
    trainer = Trainer(net_creator,env_creator,config=DEFAULT_CONFIG)

    #Train
    trainer.train(400)
