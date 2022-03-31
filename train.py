import json
import argparse
import gym

from networks import *
from env_wrapper import *
from trainer import Trainer


def run(args):
    config_file = args.config
    with open(config_file,'r') as f:
        config = json.load(f)
        f.close()
    env_name = config["env"]
    def env_creator():
        threaded_env_creator = lambda : ActionScalingWrapper(gym.make(env_name))
        return threaded_env_creator()
    
    #Networks
    policy_net_creator = actor_net
    value_net_creator = value_net

    #Build Training
    config['trainer']['seed'] = config['seed']
    trainer = Trainer(policy_net_creator,value_net_creator,env_creator,config=config['trainer'])

    #Train
    trainer.train(config['nb_steps'])
    
    
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="InvertedPendulum.json")
    args = p.parse_args()
    run(args)
