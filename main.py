from trainer import Trainer
from config import DEFAULT_TRAINER_CONFIG
from networks import *
from env_wrapper import *


if __name__ == '__main__':
    #Build creators
    def env_creator():
        #threaded_env_creator = lambda : ThreadedWrapper(JaxWrapper(gym.make('CartPole-v1')))
        threaded_env_creator = lambda : ActionScalingWrapper(gym.make('Reacher-v2'))
        return threaded_env_creator()
        #return ParallelEnv(threaded_env_creator,1)
    policy_net_creator = actor_net
    value_net_creator = value_net

    #Build Training
    trainer = Trainer(policy_net_creator,value_net_creator,env_creator,config=DEFAULT_TRAINER_CONFIG)

    #Train
    trainer.train(400)
