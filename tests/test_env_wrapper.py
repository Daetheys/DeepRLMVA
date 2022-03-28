from env_wrapper import *
import gym 

def test_threaded_wrapper():
    env = gym.make('CartPole-v0')
    th_env = ThreadedWrapper(env)
    obs = th_env.reset()
    for i in range(5):
        obs2,rew,done,_ = th_env.step(th_env.action_space.sample())
    assert obs.shape == (4,)
    assert obs2.shape == (4,)
    assert rew == 1
    assert not(done)
    th_env.close()

def test_parallel_env():
    env_creator = lambda : gym.make('CartPole-v0')
    penv = ParallelEnv(env_creator,10)
    obs = penv.reset()
    for i in range(5):
        obs2,rew,done,_ = penv.step([penv.envs[0].action_space.sample() for i in range(10)])
    assert obs.shape == (10,4)
    assert obs2.shape == (10,4)
    assert np.any(obs[0]!=obs[1])
    assert np.all(rew==1)
    assert np.all(done==False)

def test_threaded_parallel_env():
    env_creator = lambda : ThreadedWrapper(gym.make('CartPole-v0'))
    penv = ParallelEnv(env_creator,10)
    obs = penv.reset()
    for i in range(5):
        obs2,rew,done,_ = penv.step([penv.envs[0].action_space.sample() for i in range(10)])
    assert obs.shape == (10,4)
    assert obs2.shape == (10,4)
    assert np.any(obs[0]!=obs[1])
    assert np.all(rew==1)
    assert np.all(done==False)
    penv.close()